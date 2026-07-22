from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple, List, Dict

from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import try_to_load_from_cache
from unidecode import unidecode

from backend.core import config as settings
from backend.core.config import DENSE_MODEL
from backend.core.logger import log

_SHARED_MODEL: Optional[BGEM3FlagModel] = None


@contextmanager
def _gpu_session(
    *,
    task_id: Optional[str] = None,
    document_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Iterator[Any]:
    """Adquire o recurso de GPU (via gpu_resource_manager) SOMENTE ao redor do
    trabalho CUDA do BGE-M3: move o modelo para a GPU, roda a inferência, e antes
    de liberar o lease move o modelo para CPU / descarrega e limpa a VRAM.

    Fora do lock ficam: leitura de chunks, chunking, filtros, build de payload e
    upsert no Qdrant (esses acontecem em index_chunks.py, antes/depois do embed).

    Se GPU_MANAGER_ENABLED=false, roda sem coordenação (compatível com dev sem
    Redis dedicado). Fail-closed: se o Redis do manager estiver fora, a chamada
    de acquire lança GPUBackendUnavailable e a inferência NÃO ocorre."""
    if not settings.GPU_MANAGER_ENABLED:
        yield None
        return

    # Imports tardios: mantêm o embedder utilizável em contextos sem o manager
    # e evitam ciclos de import na carga do módulo.
    from backend.services.gpu_manager import get_gpu_manager
    from backend.services.bge_model_manager import get_model_manager

    manager = get_gpu_manager()
    model_manager = get_model_manager()
    with manager.acquire(
        resource=settings.GPU_RESOURCE_NAME,
        service="bge-m3",
        priority=settings.BGE_GPU_PRIORITY,
        task_id=task_id,
        document_id=document_id,
        metadata={"operation": "dense-sparse-embedding", **(metadata or {})},
    ) as lease:
        model_manager.move_to_gpu()
        try:
            yield lease
        finally:
            # 1) sincroniza CUDA, 2) modelo → CPU/unload, 3) empty_cache — ANTES de
            # liberar o lease (o lock Redis não libera VRAM sozinho).
            model_manager.release_gpu_if_configured()
            lease.ensure_valid()

# Cache LRU do embedding de query (dense + sparse), chaveado por (query, normalize).
# Permite alternar filtros sem re-embedar a mesma query — só o ANN filtrado roda.
_QUERY_CACHE: "OrderedDict[Tuple[str, bool], Tuple[List[float], Dict[str, float]]]" = OrderedDict()
_QUERY_CACHE_MAX = 256


def fold_accents(text: str) -> str:
    """Remove acentos/diacríticos (unidecode). Usado para normalizar entrada antes
    do embedding, tornando a busca lexical (sparse) robusta a acento. Aplicar de forma
    simétrica: o mesmo texto precisa ser foldado na indexação e na query."""
    return unidecode(text)


class BgeM3EmbedderService:
    """Service Adapter para o modelo BAAI/bge-m3 (Dense + Sparse).
    
    Isola a biblioteca FlagEmbedding do resto da aplicação, fornecendo métodos
    padronizados para geração de embeddings de queries e documentos em batch.
    """

    def __init__(self):
        pass

    @property
    def _model(self) -> Optional[BGEM3FlagModel]:
        return _SHARED_MODEL

    @_model.setter
    def _model(self, value: BGEM3FlagModel):
        global _SHARED_MODEL
        _SHARED_MODEL = value

    def is_loaded(self) -> bool:
        """Verifica se o modelo já está instanciado na memória global."""
        return self._model is not None

    @staticmethod
    def is_cached_locally() -> bool:
        """Verifica se os arquivos vitais do modelo já estão cacheados localmente."""
        sentinel = try_to_load_from_cache(DENSE_MODEL, "config.json")
        return sentinel is not None and sentinel is not ...

    def load_model(self, require_cache: bool = True) -> bool:
        """Carrega o modelo na memória.
        
        Args:
            require_cache: Se True, falha caso o modelo não esteja baixado localmente
                           (ideal para a API, para evitar downloads bloqueantes).
                           Se False, permite que a biblioteca faça o download automático
                           (ideal para scripts CLI como o index_chunks).
        """
        if require_cache and not self.is_cached_locally():
            log.warning(
                "Modelo '%s' não encontrado no cache local. "
                "Download automático bloqueado (require_cache=True). "
                "Execute: huggingface-cli download %s",
                DENSE_MODEL, DENSE_MODEL,
            )
            return False
            
        if self.is_loaded():
            return True
            
        try:
            log.info("Carregando modelo '%s' (use_fp16=True)...", DENSE_MODEL)
            # Ao carregar, o stdout pode ser barulhento na primeira vez, mas o logger captura o essencial
            self._model = BGEM3FlagModel(DENSE_MODEL, use_fp16=True)
            log.info("Modelo bge-m3 carregado com sucesso.")
            return True
        except Exception as exc:
            log.error("Erro ao carregar o modelo bge-m3: %s", exc)
            return False

    def embed_query(self, query: str, normalize: bool = False) -> Tuple[List[float], Dict[str, float]]:
        """Gera embedding para uma query única (busca).

        Args:
            normalize: Se True, remove acentos (unidecode) antes de embeddar. Deve casar
                       com o `normalize` usado na indexação da collection consultada.

        Returns:
            Tuple[dense_vector, lexical_weights]
        """
        cache_key = (query, normalize)
        cached = _QUERY_CACHE.get(cache_key)
        if cached is not None:
            _QUERY_CACHE.move_to_end(cache_key)
            return cached

        text = fold_accents(query) if normalize else query

        # Inferência CUDA sob o lock da GPU. O _gpu_session garante o modelo carregado
        # (move_to_gpu → ensure_loaded); a checagem vem DEPOIS de entrar na sessão
        # (o modelo pode ter sido descarregado após a task anterior).
        with _gpu_session(metadata={"operation": "query-embedding"}):
            if not self._model:
                raise RuntimeError("O modelo bge-m3 não foi carregado. Chame load_model() primeiro.")
            encoded = self._model.encode(
                [text],
                batch_size=1,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense_vector = encoded["dense_vecs"][0].tolist()
            lexical_weights = encoded["lexical_weights"][0]

        _QUERY_CACHE[cache_key] = (dense_vector, lexical_weights)
        if len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
            _QUERY_CACHE.popitem(last=False)

        return dense_vector, lexical_weights

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = False,
        *,
        task_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """Gera embeddings em batch para múltiplos documentos.

        A inferência CUDA (dense + sparse) roda sob o lock da GPU; o preparo dos
        textos (fold de acentos) fica FORA do lock. task_id/document_id são
        propagados ao gpu_resource_manager como metadados de diagnóstico.

        Args:
            normalize: Se True, remove acentos (unidecode) antes de embeddar. Deve casar
                       com o `normalize` usado na query da collection.

        Returns:
            Tuple[lista_de_dense_vectors, lista_de_lexical_weights]
        """
        # Preparo (fold de acentos) FORA do lock — não usa GPU.
        if normalize:
            texts = [fold_accents(t) for t in texts]

        # Inferência + transferência dos resultados para CPU DENTRO do lock. O
        # _gpu_session carrega/reposiciona o modelo (move_to_gpu → ensure_loaded);
        # por isso a checagem de "carregado" vem DEPOIS de entrar na sessão — o
        # modelo pode ter sido descarregado após a task anterior
        # (BGE_UNLOAD_AFTER_TASK) e é aqui que ele volta.
        with _gpu_session(
            task_id=task_id,
            document_id=document_id,
            metadata={"batch_count": len(texts), "batch_size": batch_size},
        ):
            if not self._model:
                raise RuntimeError("O modelo bge-m3 não foi carregado. Chame load_model() primeiro.")
            encoded = self._model.encode(
                texts,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            # Dense vecs já vêm como numpy array (CPU); converter aqui é a
            # "transferência dos resultados para CPU" antes de soltar o lease.
            dense_vecs = [v.tolist() for v in encoded["dense_vecs"]]
            lexical_list = encoded["lexical_weights"]

        return dense_vecs, lexical_list
        
    def get_dense_dimension(self) -> int:
        """Sonda o modelo para descobrir a dimensão real do vetor denso."""
        if not self._model:
            raise RuntimeError("O modelo bge-m3 não foi carregado.")
            
        probe = self._model.encode(
            ["dimension probe"],
            batch_size=1,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return len(probe["dense_vecs"][0])
