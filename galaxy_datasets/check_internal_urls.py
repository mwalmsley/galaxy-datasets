import os

_expected_internal_urls_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shared/internal_urls.py')
INTERNAL_URLS_EXIST = os.path.isfile(_expected_internal_urls_loc)