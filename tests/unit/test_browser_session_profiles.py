from aistudio_api.infrastructure.gateway.session import BrowserSession


def test_cleanup_profile_singletons_removes_stale_locks(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "SingletonLock").write_text("stale")
    (profile / "SingletonCookie").symlink_to("stale-cookie")
    (profile / "Preferences").write_text("{}")

    BrowserSession._cleanup_profile_singletons_sync(str(profile))

    assert not (profile / "SingletonLock").exists()
    assert not (profile / "SingletonCookie").exists()
    assert (profile / "Preferences").exists()
