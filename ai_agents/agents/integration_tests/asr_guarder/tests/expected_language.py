#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#

DEFAULT_EXPECTED_LANGUAGE = "en-US"
EXPECTED_LANGUAGE_KEY = "expected_language"


def pop_expected_language(config: dict, default: str = DEFAULT_EXPECTED_LANGUAGE) -> str:
    """Language the guarder expects in asr_result.language.

    Read from the optional top-level "expected_language" key of the test config and removed from the
    config, so the extension under test never sees it. Missing or empty -> `default` (en-US), so
    existing configs behave exactly as before. Lets a guarder run against a vendor/model that serves
    a different language (e.g. "fil-PH").
    """
    value = config.pop(EXPECTED_LANGUAGE_KEY, None)
    return value if isinstance(value, str) and value else default
