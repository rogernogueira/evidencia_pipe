"""Testes do MinerUDocumentParser (§32: títulos/headings, section_path, ordem,
tabelas, listas, referências, header/footer repetidos, fallback markdown)."""

from backend.indexing.chunk_models import (
    BLOCK_FIGURE_CAPTION,
    BLOCK_HEADING,
    BLOCK_LIST,
    BLOCK_REFERENCE,
    BLOCK_TABLE,
)
from backend.indexing.document_blocks import MinerUDocumentParser


def _title(text, level=1):
    return {"type": "title", "content": {"title_content": [{"type": "text", "content": text}], "level": level}, "bbox": [0, 0, 1, 1]}


def _para(text):
    return {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": text}]}}


def test_title_and_section_path_hierarchy():
    data = [[
        _title("Capítulo 1", 1),
        _title("Seção 1.1", 2),
        _para("conteúdo da seção"),
        _title("Capítulo 2", 1),
        _para("outro conteúdo"),
    ]]
    parser = MinerUDocumentParser()
    blocks, src = parser.parse_json(data)
    assert src == "mineru_json"
    body = [b for b in blocks if b.block_type != BLOCK_HEADING]
    assert body[0].section_path == ["Capítulo 1", "Seção 1.1"]
    assert body[1].section_path == ["Capítulo 2"]
    # order preservada
    assert [b.order_index for b in blocks] == sorted(b.order_index for b in blocks)


def test_heading_level_and_document_title():
    data = [[_title("Relatório X", 2), _para("intro")]]
    parser = MinerUDocumentParser()
    blocks, _ = parser.parse_json(data)
    heading = next(b for b in blocks if b.block_type == BLOCK_HEADING)
    assert heading.heading_level == 2
    assert parser.document_title == "Relatório X"


def test_list_parsing():
    data = [[{
        "type": "list",
        "content": {"list_type": "text_list", "list_items": [
            {"item_type": "text", "item_content": [{"type": "text", "content": "item um"}]},
            {"item_type": "text", "item_content": [{"type": "text", "content": "item dois"}]},
        ]},
    }]]
    blocks, _ = MinerUDocumentParser().parse_json(data)
    lst = next(b for b in blocks if b.block_type == BLOCK_LIST)
    assert "item um" in lst.text and "item dois" in lst.text
    assert lst.metadata["item_count"] == 2


def test_table_parsing_html_to_markdown():
    data = [[{
        "type": "table",
        "content": {"html": "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>",
                    "table_caption": [{"type": "text", "content": "Tabela 1"}]},
    }]]
    blocks, _ = MinerUDocumentParser().parse_json(data)
    tbl = next(b for b in blocks if b.block_type == BLOCK_TABLE)
    assert "Tabela 1" in tbl.text
    assert tbl.metadata["table_caption"] == "Tabela 1"


def test_references_section_detection():
    data = [[
        _title("Referências", 1),
        _para("AUTOR, A. Título. Editora, 2020."),
    ]]
    blocks, _ = MinerUDocumentParser().parse_json(data)
    ref = next(b for b in blocks if b.block_type == BLOCK_REFERENCE)
    assert "AUTOR" in ref.text


def test_repeated_headers_removed_and_page_numbers():
    def _header(text):
        return {"type": "page_header", "content": {"page_header_content": [{"type": "text", "content": text}]}}

    def _pagenum(n):
        return {"type": "page_number", "content": {"page_number_content": [{"type": "text", "content": str(n)}]}}

    pages = []
    for i in range(4):
        pages.append([_header("CABEÇALHO REPETIDO"), _para(f"corpo {i}"), _pagenum(i + 1)])
    parser = MinerUDocumentParser(remove_repeated_headers=True)
    blocks, _ = parser.parse_json(pages)
    assert parser.removed_header_blocks == 4
    assert parser.removed_page_number_blocks == 4
    assert all("CABEÇALHO" not in b.text for b in blocks)


def test_markdown_fallback():
    md = "# Título\n\nUm parágrafo.\n\n- item a\n- item b\n"
    parser = MinerUDocumentParser()
    blocks, src = parser.parse_markdown(md)
    assert src == "markdown"
    assert any(b.block_type == BLOCK_HEADING for b in blocks)
    assert any(b.block_type == BLOCK_LIST for b in blocks)


def test_figure_caption_from_image():
    data = [[{"type": "image", "content": {"image_source": {"path": "images/x.jpg"}, "content": "diagrama de fluxo"}}]]
    blocks, _ = MinerUDocumentParser(images_base_uri="minio://b/imgs").parse_json(data)
    fig = next(b for b in blocks if b.block_type == BLOCK_FIGURE_CAPTION)
    assert fig.text == "diagrama de fluxo"
    assert fig.metadata["image_uri"] == "minio://b/imgs/x.jpg"
