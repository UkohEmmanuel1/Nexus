import pytest

from src.agents.file_search import FileSearch


def test_index_skips_dependencies_and_secret_files(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dependency.py").write_text("private dependency")
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / ".env").write_text("SECRET=private")
    search = FileSearch(str(tmp_path))
    search.index_directory(patterns=["*"])
    assert list(search.index) == ["main.py"]


def test_read_and_index_cannot_escape_root(tmp_path):
    search = FileSearch(str(tmp_path))
    with pytest.raises(ValueError, match="configured root"):
        search.read_file("../outside.txt")
    with pytest.raises(ValueError, match="configured root"):
        search.index_directory(str(tmp_path.parent))
