"""Target-independent called-number fields from OCR, pixel spacing and local labels.

Numbers are never supplied by a user's waiting target or a free-form model answer.
Rows stay associated with their nearby heading rather than pooling screen text.
"""
import re
import unicodedata
import cv2
import numpy as np


def compact(text):
    return re.sub(r'[\s_\-‐‑–:：]+','',unicodedata.normalize('NFKC',text).upper())


def heading_role(text):
    text=compact(text)
    if re.search('等待|等候|候诊|候診|候餐|未叫|取号|取號|排队|排隊|WAITING|PREPARING|PENDING|NOTREADY',text):return 'waiting'
    if re.search('取餐|取药|取藥|叫号|叫號|当前号|當前號|正在服务|NOWSERVING|NOWCALLING|PICKUP|READYFOR|CALLEDNUMBER',text):return 'called'
    if re.search('窗口|柜台|櫃台|诊室|診室|COUNTER|DESK|ROOM|LOKET|SCHALTER',text):return 'counter'
    if re.search('价格|價格|金额|金額|汇率|匯率|利率|PRICE|SELL|BUY|CNY|EUR|USD|VALAS|SALDO|BUNGA',text):return 'other'
    if re.search('号码|號碼|票号|票號|QUEUENUMBER|TICKETNUMBER|NOMOR|ANTRIAN|ANTREAN|UWNUMMER',text) or text=='NUMBER':return 'queue_label'
    return None


def rect(entry):
    points=np.asarray(entry['box'],float)
    return np.r_[points.min(axis=0),points.max(axis=0)]


def overlap(a,b):
    size=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    intersection=float(np.prod(size));area=float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))
    return intersection/max(1,area-intersection)


def numeric_row(text):
    value=unicodedata.normalize('NFKC',text).upper()
    # A numeric hyphen/slash can join ticket and counter columns. It is not a
    # letter prefix (C-002); do not treat both fields as called tickets.
    if re.search(r'(?<![A-Z0-9])\d+\s*[-/]\s*\d+',value):return False
    return bool(re.search(r'\d',value) and re.fullmatch(r'[A-Z0-9\s,，、;/\-‐‑–]+',value))


def split_numeric_row(image,entry):
    """Split concatenated digits only where visible glyph gaps support boundaries."""
    text=entry['text']
    if not re.fullmatch(r'[0-9]{3,}',text):return [entry]
    box=np.asarray(entry['box'],np.float32)
    width=round(float(np.linalg.norm(box[1]-box[0])));height=round(float(np.linalg.norm(box[3]-box[0])))
    if min(width,height)<8:return [entry]
    dest=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    transform=cv2.getPerspectiveTransform(box,dest)
    crop=cv2.warpPerspective(image,transform,(width,height))
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    border=np.r_[binary[0],binary[-1],binary[:,0],binary[:,-1]]
    if np.mean(border)>127:binary=255-binary
    occupied=(binary[height//5:4*height//5]>0).sum(0)>.10*height
    edges=np.diff(np.r_[False,occupied,False].astype(np.int8))
    spans=[(int(a),int(b)) for a,b in zip(np.where(edges==1)[0],np.where(edges==-1)[0]) if b-a>=max(2,height*.025)]
    # Every recognized digit must align with one visible glyph; never cut by an
    # assumed ticket length or by a substring matching the desired number.
    if len(spans)!=len(text):return [entry]
    gaps=np.array([spans[i+1][0]-spans[i][1] for i in range(len(spans)-1)])
    threshold=max(.6*np.median([b-a for a,b in spans]),2*np.median(gaps))
    breaks=[i+1 for i,gap in enumerate(gaps) if gap>threshold]
    if not breaks:return [entry]
    inverse=cv2.getPerspectiveTransform(dest,box);result=[]
    for a,b in zip([0]+breaks,breaks+[len(text)]):
        local=np.array([[[spans[a][0],0],[spans[b-1][1],0],[spans[b-1][1],height-1],[spans[a][0],height-1]]],np.float32)
        original=cv2.perspectiveTransform(local,inverse)[0]
        result.append(dict(entry,text=text[a:b],box=original.tolist(),number_source='ocr_pixel_gap_split',original_text=text))
    return result


def _heading_distance(row,heading,allow_same_line=True):
    # Compare along the heading's baseline. Camera roll/perspective can make a
    # lower row move right in image coordinates while staying in the same panel.
    corners=np.asarray(heading['box'],float)
    axis=corners[1]-corners[0];axis=axis/max(1,np.linalg.norm(axis))
    basis=np.array([axis,[-axis[1],axis[0]]]).T
    def in_heading(entry):
        points=np.asarray(entry['box'],float)@basis
        return np.r_[points.min(0),points.max(0)]
    r,h=in_heading(row),in_heading(heading);rh=max(1,r[3]-r[1]);hh=max(1,h[3]-h[1])
    rc=(r[:2]+r[2:])/2;hc=(h[:2]+h[2:])/2
    horizontal=max(0,min(r[2],h[2])-max(r[0],h[0]))/max(1,min(r[2]-r[0],h[2]-h[0]))
    # A number can follow a heading on the same line or below it in a column.
    if abs(rc[1]-hc[1])<.65*max(rh,hh) and rc[0]>hc[0]:
        if not allow_same_line:return None
        return max(0,r[0]-h[2])/max(rh,hh)+.1 if r[0]-h[2]<6*max(rh,hh) else None
    if horizontal>=.45 and rc[1]>hc[1] and rc[1]-hc[1]<6*max(rh,hh):
        return (rc[1]-hc[1])/max(rh,hh)
    return None


def classify_rows(entries):
    from .tasks import identifiers
    headers=[dict(e,queue_role=heading_role(e['text'])) for e in entries if e['score']>=.65 and heading_role(e['text'])]
    output=[]
    for entry in entries:
        e=dict(entry,queue_verification_attempted=True,queue_role='other')
        own_role=heading_role(e['text'])
        if own_role in ('counter','waiting','other','queue_label'):
            # A labelled counter/price/waiting field cannot inherit a called role
            # from an adjacent column (e.g. "Loket 2" below a large ticket header).
            e['queue_role']=own_role;output.append(e);continue
        if e['score']>=.7 and numeric_row(e['text']):
            choices=[(distance,heading) for heading in headers if (distance:=_heading_distance(e,heading)) is not None]
            if choices:
                _,heading=min(choices,key=lambda value:value[0]);role=heading['queue_role']
                if role=='queue_label':
                    # Generic queue-number wording alone can also label a ticket
                    # dispenser. Require a nearby service-counter heading above.
                    active=any(h['queue_role']=='counter' and _heading_distance(heading,h,allow_same_line=False) is not None for h in headers)
                    active=active or any(h['queue_role']=='counter' and not identifiers(h['text'])
                        and _heading_distance(h,heading) is not None
                        and _heading_distance(h,heading,allow_same_line=False) is None for h in headers)
                    role='called' if active else 'unconfirmed'
                elif role=='counter' and identifiers(heading['text']) and _heading_distance(e,heading,allow_same_line=False) is not None:
                    # The counter number is already in its own label; the separate
                    # number underneath is that counter's current ticket field.
                    role='called'
                e.update(queue_role=role,queue_heading=heading['text'],queue_heading_box=heading['box'])
        output.append(e)
    return output


def read_queue(image,ocr):
    entries=ocr(image);height,width=image.shape[:2]
    initial=classify_rows(entries)
    proposals=[]
    # Re-read context when a digit row lost its heading or merged its whitespace.
    # Regions come from OCR boxes, never target IDs or hand-labelled screens.
    for entry in initial:
        if entry['score']<.7 or not re.fullmatch(r'[0-9]{2,}',entry['text']):continue
        if len(entry['text'])<7 and entry['queue_role'] not in ('other','unconfirmed'):continue
        points=np.asarray(entry['box']);x1,y1,x2,y2=rect(entry)
        h=min(np.linalg.norm(points[0]-points[3]),np.linalg.norm(points[1]-points[2]));pad=.5*(x2-x1)
        proposed=np.array([max(0,round(x1-pad)),max(0,round(y1-4*h)),min(width,round(x2+pad)),min(height,round(y2+2*h))])
        if any(overlap(proposed,p)>.5 for p in proposals):continue
        proposals.append(proposed)
        if len(proposals)==4:break
    for x1,y1,x2,y2 in proposals:
        for raw in ocr(image[y1:y2,x1:x2].copy()):
            shifted=dict(raw,box=(np.asarray(raw['box'])+[x1,y1]).tolist(),ocr_context_refined=True)
            matches=[i for i,e in enumerate(entries) if overlap(rect(e),rect(shifted))>.6]
            if matches:entries[matches[0]]=shifted
            else:entries.append(shifted)
    split=[part for e in entries for part in split_numeric_row(image,e)]
    return classify_rows(split)


def called_in_audio(text):
    """Return called identifiers, excluding destinations/counters and negations."""
    from .tasks import identifiers
    result=[]
    for clause in re.split(r'[，,。；;！？!?]+',text):
        if re.search('未|还没|還沒|尚未|不是|并非|並非|取消|过号|過號|跳过|跳過|等待|排队|排隊|请问|請問|有没有|有沒有|是否|哪里|哪裡',clause):continue
        spoken=None
        imperative=re.search(r'(?:请|請)(.+)',clause)
        if imperative and re.search('前往|到|至|去|就诊|就診|取餐|取药|取藥|办理|辦理',imperative[1]):
            spoken=re.split('前往|到|至|去|就诊|就診|取餐|取药|取藥|办理|辦理',imperative[1],maxsplit=1)[0]
            if not identifiers(spoken):
                before=clause[:imperative.start()]
                if not re.search('窗口|柜台|櫃台|诊室|診室',before):spoken=before
        else:
            label=re.search(r'(?:叫到|叫号|叫號|呼叫|NOW\s+(?:SERVING|CALLING))\s*(.+)',clause,re.I)
            if label:spoken=re.split('前往|到|至|去|窗口|柜台|櫃台|COUNTER|DESK',label[1],maxsplit=1,flags=re.I)[0]
        if spoken and not re.search(r'(?<![A-Z0-9])\d+\s*[-/]\s*\d+',spoken.upper()):
            result.extend(identifiers(spoken))
    return list(dict.fromkeys(result))
