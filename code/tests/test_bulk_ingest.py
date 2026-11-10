from app import bulk_ingest


def test_dry_run_lists_supported_files_and_skips_others(tmp_path, capsys):
    (tmp_path / "a.md").write_text("# A\n\nhello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")
    (tmp_path / "c.csv").write_text("skip", encoding="utf-8")

    code = bulk_ingest.run(tmp_path, dry_run=True)
    output = capsys.readouterr().out

    assert code == 0
    assert "a.md" in output
    assert "b.txt" in output
    assert "c.csv" not in output
    assert "2/2 files processed" in output


def test_failures_are_reported_and_other_files_continue(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    (tmp_path / "b.md").write_text("world", encoding="utf-8")

    def fake_index_file(path):
        if path.name == "a.md":
            raise ValueError("boom")
        return 1, ["id"]

    monkeypatch.setattr(bulk_ingest, "index_file", fake_index_file)

    code = bulk_ingest.run(tmp_path)
    captured = capsys.readouterr()

    assert code == 1
    assert "1/2 files processed" in captured.out
    assert "a.md: FAILED ValueError: boom" in captured.err
