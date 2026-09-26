import os

from tfcta.rq_auth import load_rqdata_env, rqdata_credentials


def test_load_rqdata_env_fills_blank_names_without_override(tmp_path, monkeypatch):
    monkeypatch.delenv("RQDATAC_LICENSE", raising=False)
    monkeypatch.delenv("RQDATAC_USERNAME", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)
    monkeypatch.setenv("RQDATAC_USERNAME", "already-set")
    path = tmp_path / "rqdata.env"
    path.write_text(
        "\n".join([
            "# comment",
            "RQDATAC_LICENSE=",
            "export RQDATAC_USERNAME='from-file'",
            'RQDATAC_PASSWORD="secret"',
            "",
        ]),
        encoding="utf-8",
    )

    load_rqdata_env(path)

    assert os.environ["RQDATAC_USERNAME"] == "already-set"
    assert os.environ["RQDATAC_PASSWORD"] == "secret"
    assert "RQDATAC_LICENSE" not in os.environ


def test_rqdata_credentials_prefers_license(tmp_path, monkeypatch):
    monkeypatch.delenv("RQDATAC_LICENSE", raising=False)
    monkeypatch.delenv("RQDATAC_USERNAME", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)
    path = tmp_path / "rqdata.env"
    path.write_text(
        "RQDATAC_LICENSE=token\nRQDATAC_USERNAME=user\nRQDATAC_PASSWORD=pw\n",
        encoding="utf-8",
    )

    assert rqdata_credentials(path) == ("token",)


def test_rqdata_credentials_uses_password_when_license_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("RQDATAC_LICENSE", raising=False)
    monkeypatch.delenv("RQDATAC_USERNAME", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)
    path = tmp_path / "rqdata.env"
    path.write_text("RQDATAC_USERNAME=user\nRQDATAC_PASSWORD=pw\n", encoding="utf-8")

    assert rqdata_credentials(path) == ("user", "pw")


def test_missing_env_file_leaves_credentials_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("RQDATAC_LICENSE", raising=False)
    monkeypatch.delenv("RQDATAC_USERNAME", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)

    assert rqdata_credentials(tmp_path / "absent.env") is None
