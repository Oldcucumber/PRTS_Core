"""Explicit control vocabulary first; open-ended language falls back to the local model.

Rules refer to task semantics, never to a particular route, queue number or fixture.
"""
import re
import unicodedata
from .tasks import identifiers


def normalize(text):
    return unicodedata.normalize('NFKC', text).translate(str.maketrans(
        '請號車來綠紅燈開導航繼續暫這個時剛幫換櫃臺輛進達線轉',
        '请号车来绿红灯开导航继续暂这个时刚帮换柜台辆进达线转'))


def route(text, current_wait=None, has_candidates=False):
    t = re.sub(r'\s+', '', normalize(text))
    def result(action, kind='', target='', **extra):
        return dict(action=action, kind=kind, target=target, source='semantic_rules', **extra)
    if current_wait and re.fullmatch(r'(?:叫到时|到时|到了|来了)(?:再)?(?:提醒|告诉|通知)我[。！!，,]?',t):
        return result('continue_wait')

    if re.search('(不要|别).{0,2}(停止|暂停).{0,5}(导航|指路|引导|方向提示)',t):
        return result('navigate')
    if re.search('(不要|不用|别|停止|暂停|取消).{0,8}(导航|指路|带路|引导|方向提示|提示方向|报方向|方向指引)',t):
        return result('stop')

    if has_candidates:
        selection = re.fullmatch(r'(?:选|选择|就去|就是|确认|第)?([一二三四五12345])(?:个|处|家)?[。！!，,]?', t)
        if selection:
            index = int(selection[1]) if selection[1].isdigit() else '一二三四五'.index(selection[1])+1
            return result('confirm_destination', index=index)
        if re.fullmatch(r'(确认|就这个|就是这个|好的|可以)[。！!，,]?', t):
            return result('confirm_destination', index=0)

    destination = re.search(r'(?:带我去|导航到|我想要去|我想去|我要去|带我到|前往)(.+)', t)
    if destination and not re.search(r'不想去|不要去|不是去', t):
        pieces = re.split(r'[，,。；;]|到了(?:以后|之后)?|到站后', destination[1], maxsplit=1)
        name = re.sub(r'[吧啊呀呢？?！!。]+$', '', pieces[0])
        arrival_wait = route(pieces[1], current_wait) if len(pieces)>1 else None
        if name:
            return result('destination', query=name,
                          arrival_wait=arrival_wait if arrival_wait and arrival_wait['action']=='wait' else None)

    waiting = bool(re.search('等待|等候|提醒|通知|留意|盯着|叫到|到站|来了|来时|到了|到时|等.{0,15}(路|号|车|灯|线)', t))
    if not waiting and re.search('能不能走|可以.{0,18}(走|通行|通过).{0,2}[吗么]?|能否.{0,8}(走|通行)|能走吗',t) and not re.search('开始导航|恢复导航|开启导航',t):
        return result('ask')
    if re.search('(取消|撤销).{0,4}(导航|指路|引导)', t):
        return result('stop')
    if re.search('取消|撤销|不等|别等|停止等待|停止等候|不用.{0,8}(提醒|通知)|不要.{0,8}(提醒|通知)', t):
        return result('cancel')
    if re.search('停下|停一停|暂停|停止.{0,5}(导航|指路|引导|走)|别.{0,4}(走|指路|导航)|不要.{0,5}(走|指路|导航)', t):
        return result('stop')
    if not waiting and re.search('开始|继续|恢复|开启|带我|帮我|往前|向前', t) and re.search('导航|带路|指路|引导|走|前进', t):
        return result('navigate')

    kind = None
    # Explicit calling/queue semantics take precedence over a incidental platform name.
    if re.search('叫号|叫到|排号|排队|取餐|就诊|候诊|挂号|窗口|柜台|我的号', t): kind = 'number'
    elif re.search('公交|公车|巴士|公共汽车|路车|路的车', t): kind = 'bus'
    elif re.search('地铁|列车|火车|号线|轻轨', t): kind = 'train'
    elif re.search('红绿灯|信号灯|绿灯|红灯|黄灯', t): kind = 'light'
    modify = bool(re.search('改成|换成|改为|改等|换等|换一下', t))
    if modify and kind is None and current_wait: kind = current_wait['kind']
    if kind and (waiting or modify):
        ids = identifiers(t)
        if kind == 'light':
            colors = re.findall('(绿|红|黄)灯', t)
            target = colors[-1] if colors else ''
        else:
            # Prefer the number explicitly attached to a route/line/queue noun.
            suffix = '路' if kind == 'bus' else '号线|线' if kind == 'train' else '号'
            chunks = re.findall(r'([A-Za-z夜快特临专臨專]*[0-9零〇一幺二两三四五六七八九十百千]+[A-Za-z]*)(?:' + suffix + ')', t)
            choices = [identifiers(x)[0] for x in chunks if identifiers(x)]
            if not choices: choices = ids
            # A modification can mention the previous number before the new number.
            target = choices[-1] if modify and choices else choices[0] if len(set(choices)) == 1 else ''
        direction = re.search(r'(?:开往|往|去往)(.+?)(?:方向|的|[，,。]|$)', t) if kind in ('bus','train') else None
        return result('wait', kind, target, direction=direction[1] if direction else '')
    if waiting and re.search('等|提醒|通知|留意', t):
        return result('wait', '', '', needs_clarification=True)
    # Clear perception queries should never create a waiting task.
    if re.search('什么|哪里|哪儿|在哪|写的|写着|读一下|读出来|看一下|看看|能不能|可以|有没有|前面|眼前|周围', t):
        return result('ask')
    return None


def read_text_request(text):
    t = normalize(text)
    return bool(re.search('写|读|文字|牌|标识|告示|标志|内容', t))


def ordered_ocr(entries, threshold=.80):
    """Group boxes into rows by vertical overlap, then preserve their left-to-right order.

    Keep separate text regions separated by /; do not infer scope or permissions.
    """
    boxes = []
    for item in entries:
        if item['score'] < threshold: continue
        xs, ys = zip(*item['box'])
        boxes.append((min(xs), min(ys), max(xs), max(ys), item['text']))
    rows = []
    for box in sorted(boxes, key=lambda b: b[1]):
        for row in rows:
            overlap = min(box[3], row['bottom']) - max(box[1], row['top'])
            if overlap > .45 * min(box[3]-box[1], row['bottom']-row['top']):
                row['boxes'].append(box); row['top'] = min(row['top'], box[1]); row['bottom'] = max(row['bottom'], box[3]); break
        else: rows.append(dict(top=box[1], bottom=box[3], boxes=[box]))
    return [' / '.join(b[4] for b in sorted(row['boxes'])) for row in sorted(rows, key=lambda r:r['top'])]
