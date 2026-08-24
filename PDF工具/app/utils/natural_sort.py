import re


def natural_sort_key(s: str) -> list:
    """生成自然排序键，使 'chapter2' < 'chapter10'"""
    parts = re.split(r'(\d+)', s.lower())
    result = []
    for part in parts:
        if part.isdigit():
            result.append((0, int(part)))
        else:
            result.append((1, part))
    return result


def natural_sort(items: list[str]) -> list[str]:
    """对字符串列表进行自然排序（原地排序）"""
    items.sort(key=natural_sort_key)
    return items