from medpmc_initial_screening.licenses import license_is_allowed, normalize_license


def test_license_normalization():
    assert normalize_license("CC BY-NC-SA") == "CC BY-NC-SA"
    assert normalize_license("https://creativecommons.org/licenses/by/4.0/") == "CC BY"
    assert normalize_license("CC BY-NC-ND") == "CC BY-NC-ND"


def test_default_filter():
    assert license_is_allowed("CC BY")
    assert license_is_allowed("CC BY-NC")
    assert not license_is_allowed("CC BY-ND")
    assert not license_is_allowed("CC BY-NC-ND")
    assert not license_is_allowed("NO-CC CODE")
