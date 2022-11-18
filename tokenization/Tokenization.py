import re

def fix_accents(regularized: str):
    """
    Replace gravis by acutus, removes second acutus if the following word is enclitic
    """
    
    
    if '\u0300' in regularized:  # check for gravis and replace with acutus
        return regularized.replace('\u0300', '\u0301')

    # Match the whole word, containing acutus/perispomenus up until the second acutus in group 1, and the rest, excluding the acutus in group 2.
    double_acutus_pattern = '([^\u0301\u0342]*[\u0301\u0342][^\u0301\u0342]*)\u0301([^\u0301\u0342]*)'
    m = re.match(double_acutus_pattern, regularized)

    if m:
        return re.sub(double_acutus_pattern, m.group(1) + m.group(2), regularized)

    return regularized
