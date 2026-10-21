from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as E

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'figures' / 'fig1_gblcss_overview_drawio_v20_method_aligned.drawio'
OUT = ROOT / 'figures' / 'revised_20260914'
WORK = ROOT / '.drawio-tmp' / 'submission_figures_20260914'
INK, MUTED, LINE = '#26323D', '#586774', '#AAB7C2'
BLUE, ORANGE = '#397DA9', '#D55E00'
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#999999']
old = E.parse(SOURCE)
old_cells = {c.get('id'): c for c in old.findall('.//mxCell')}
doc = E.Element('mxfile', host='Electron', agent='draw.io', version='28.0.0')
diagram = E.SubElement(doc, 'diagram', id='gblcss-v21', name='GB-LCSS overview')
model = E.SubElement(diagram, 'mxGraphModel', dx='1400', dy='1000', grid='0', gridSize='10',
    guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='0', pageScale='1',
    pageWidth='1400', pageHeight='1000', math='1', shadow='0', background='#FFFFFF')
cells = E.SubElement(model, 'root')
E.SubElement(cells, 'mxCell', id='0')
E.SubElement(cells, 'mxCell', id='1', parent='0')
used = {'0','1'}
counter = 0


def vertex(id, x, y, w, h, value='', style=''):
    assert id not in used, id
    used.add(id)
    c = E.SubElement(cells, 'mxCell', id=id, parent='1', value=value, vertex='1', style=style)
    E.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as':'geometry'})
    return id


def text(id,x,y,w,h,value,size=20,bold=False,color=INK,align='center'):
    return vertex(id,x,y,w,h,value,
        f'text;html=1;whiteSpace=wrap;overflow=visible;rounded=0;fillColor=none;strokeColor=none;'
        f'fontFamily=Arial;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};'
        f'align={align};verticalAlign=middle;spacing=0;')


def box(id,x,y,w,h,value='',fill='#FFFFFF',stroke=LINE,size=20,bold=False,radius=10,dashed=False):
    return vertex(id,x,y,w,h,value,
        f'rounded=1;absoluteArcSize=1;arcSize={radius};whiteSpace=wrap;html=1;'
        f'fillColor={fill};strokeColor={stroke};strokeWidth=1.2;'
        f'fontFamily=Arial;fontSize={size};fontColor={INK};fontStyle={1 if bold else 0};spacing=5;'
        f'dashed={int(dashed)};')


def circle(id,x,y,d,fill='none',stroke=LINE,lw=1.2,value='',size=20,dashed=False):
    return vertex(id,x,y,d,d,value,
        f'ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={lw};'
        f'fontFamily=Arial;fontSize={size};fontColor={INK};dashed={int(dashed)};')


def anchor(x,y):
    global counter
    counter += 1
    return vertex(f'anchor-{counter}',x-.05,y-.05,.1,.1,style='fillColor=none;strokeColor=none;opacity=0;')


def edge(id,start,end,waypoints=(),color=INK,lw=1.5,dashed=False,arrow=True):
    source = anchor(*start)
    target = anchor(*end)
    assert id not in used
    used.add(id)
    c = E.SubElement(cells,'mxCell',id=id,parent='1',source=source,target=target,edge='1',
        style=f'edgeStyle=none;rounded=0;html=1;strokeColor={color};strokeWidth={lw};'
              f'dashed={int(dashed)};endArrow={"blockThin" if arrow else "none"};endFill=1;endSize=7;')
    g=E.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
    if waypoints:
        a=E.SubElement(g,'Array',**{'as':'points'})
        for x,y in waypoints:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
    return c


def panel(id,x,y,w,h,n,title):
    box(id,x,y,w,h,stroke='#687985',radius=14)
    vertex(id+'-header',x+1,y+1,w-2,44,style='rounded=0;fillColor=#F1F4F6;strokeColor=none;')
    circle(id+'-number',x+14,y+8,28,fill=INK,stroke=INK,value=str(n),size=20)
    oldnum=cells[-1]
    oldnum.set('style',oldnum.get('style').replace(f'fontColor={INK}','fontColor=#FFFFFF'))
    text(id+'-title',x+54,y+7,w-66,32,title,size=22,bold=True,align='left')


def geometries(c):
    g=c.find('mxGeometry')
    if g is None:return None
    return tuple(float(g.get(k,0)) for k in ['x','y','width','height'])


def clone_group(ids, prefix, source_bounds, target_bounds, font_size=None):
    sx,sy,sw,sh=source_bounds;tx,ty,tw,th=target_bounds
    scale=min(tw/sw,th/sh)
    ox=tx+(tw-sw*scale)/2;oy=ty+(th-sh*scale)/2
    for id in ids:
        src=old_cells[id];c=copy.deepcopy(src);new_id=prefix+'-'+id
        assert new_id not in used
        used.add(new_id);c.set('id',new_id);c.set('parent','1')
        c.attrib.pop('source',None);c.attrib.pop('target',None)
        g=c.find('mxGeometry')
        if g is not None:
            if 'x' in g.attrib:g.set('x',str(ox+(float(g.get('x'))-sx)*scale))
            if 'y' in g.attrib:g.set('y',str(oy+(float(g.get('y'))-sy)*scale))
            for k in ['width','height']:
                if k in g.attrib:g.set(k,str(float(g.get(k))*scale))
            for point in g.findall('.//mxPoint'):
                point.set('x',str(ox+(float(point.get('x'))-sx)*scale))
                point.set('y',str(oy+(float(point.get('y'))-sy)*scale))
        style=c.get('style','')
        style=style.replace('fontFamily=Helvetica','fontFamily=Arial')
        if font_size is not None and c.get('value'):
            import re
            style=re.sub(r'fontSize=[^;]+',f'fontSize={font_size}',style)
        c.set('style',style)
        cells.append(c)


def scatter_ids(bounds, max_size=15):
    x0,y0,x1,y1=bounds
    found=[]
    for id,c in old_cells.items():
        g=geometries(c)
        if not g or 'ellipse;' not in c.get('style',''):continue
        x,y,w,h=g
        if 0<w<=max_size and x0<=x+w/2<=x1 and y0<=y+h/2<=y1:found.append(id)
    return found


def plotbars(prefix,x,y,w,h,values,title='',axis_one=False):
    edge(prefix+'-x',(x,y+h),(x+w,y+h),color=MUTED,lw=.9,arrow=False)
    edge(prefix+'-y',(x,y+h),(x,y),color=MUTED,lw=.9,arrow=False)
    bw=w/7.8;gap=w/5
    for i,v in enumerate(values):
        bx=x+gap*(i+.2)
        if v>0:vertex(prefix+f'-bar{i}',bx,y+h*(1-v),bw,h*v,
            style=f'rounded=0;fillColor={COLORS[i]};strokeColor={INK};strokeWidth=.5;')
        text(prefix+f'-tick{i}',bx-3,y+h+5,bw+6,21,str(i+1),size=17)
    if axis_one:
        text(prefix+'-one',x-22,y-8,17,20,'1',size=17,color=MUTED)
        text(prefix+'-zero',x-22,y+h-9,17,20,'0',size=17,color=MUTED)
    if title:text(prefix+'-title',x-15,y-42,w+30,28,title,size=19)


def mlp(prefix,x,y,w=74,h=66):
    left=[y+h*i/3 for i in range(4)]
    right=[y+h*i/4 for i in range(5)]
    for i,ly in enumerate(left):
        for j,ry in enumerate(right):edge(prefix+f'-wire{i}-{j}',(x+6,ly),(x+w-6,ry),color='#A7BBCD',lw=.65,arrow=False)
    for i,ly in enumerate(left):circle(prefix+f'-left{i}',x-5,ly-5,10,fill='white',stroke=BLUE,lw=1)
    for i,ry in enumerate(right):circle(prefix+f'-right{i}',x+w-5,ry-5,10,fill='white',stroke=BLUE,lw=1)


def feature(prefix,x,y):
    for i in range(5):vertex(prefix+str(i),x+12*i,y,12,16,style=f'fillColor={"#6B90BA" if i%2==0 else "#FFFFFF"};strokeColor=#A4B4C3;strokeWidth=.7;')


def create():
    panel('stage1',12,12,602,397,1,'Label-free representation learning')
    panel('stage2',636,12,752,397,2,'Granular-ball partitioning')
    panel('stage3',12,443,838,459,3,'Consensus-weighted soft targets')
    panel('stage4',874,443,514,459,4,'MLP training and graph-free inference')

    text('input-title',27,79,128,27,'Input graph',bold=True)
    clone_group([f'l{i}' for i in range(20,32)]+[f'e{i}' for i in range(32,40)],
                'input',(40,76,116,106),(34,117,112,104))
    text('input-xa',35,321,108,26,r'\(X, A\)',size=20)
    for k,x in enumerate([39,98]):
        for i in range(4):
            for j in range(4):
                filled=((i+j)%3==0) if k==0 else (i!=j and (i+j)%2==1)
                vertex(f'input-matrix-{k}-{i}-{j}',x+j*10,252+i*10,10,10,
                    style=f'fillColor={"#568AB3" if filled else "#FFFFFF"};strokeColor=#B7C5D0;strokeWidth=.6;')
    text('views-title',171,79,133,27,'Two views',bold=True)
    box('view1',177,119,120,88,fill='#FAFBFC')
    box('view2',177,246,120,88,fill='#FAFBFC')
    view1=[f'l{i}' for i in range(94,106) if i!=98]+[f'e{i}' for i in range(106,114)]
    view2=[f'l{i}' for i in range(114,126) if i!=121]+[f'e{i}' for i in range(126,134)]
    clone_group(view1,'aug1',(233,95,71,68),(203,125,73,73))
    clone_group(view2,'aug2',(233,247,71,68),(203,252,73,73))
    edge('view-in-1',(143,218),(177,163),[(160,218),(160,163)])
    edge('view-in-2',(160,218),(177,290),[(160,290)])
    box('encoder',328,132,116,106,'Graph encoder<br>\\(g_\\psi\\)<br>(GRACE)',fill='#EAF2F7',stroke=BLUE,size=19,bold=True)
    box('projection',328,293,116,48,'Projection<br>head',fill='#F5F7F8',size=18)
    edge('aug1-encoder',(297,163),(328,163))
    edge('aug2-encoder',(297,290),(328,205),[(313,290),(313,205)])
    edge('encoder-projection',(385,238),(385,293))
    text('frozen-title',462,78,143,30,r'Frozen \(\bar H\)',size=20,bold=True)
    box('frozen',475,118,120,125,fill='#FFFFFF',stroke='#AABBC9',radius=14)
    cloud=scatter_ids((600,110,800,333))
    clone_group(cloud,'frozen-cloud',(600,110,200,223),(480,125,110,109))
    edge('encoder-frozen',(444,179),(475,179))
    text('contrastive-title',463,250,145,25,'Contrastive loss',size=18)
    for k,x in enumerate([470,557]):
        for i in range(3):
            for j in range(2):vertex(f'z-{k}-{i}-{j}',x+j*13,302+i*11,13,11,
                style=f'fillColor={"#6B90BA" if (i+j)%2==0 else "#FFFFFF"};strokeColor=#B7C5D0;strokeWidth=.5;')
        text(f'z-title-{k}',x-4,275,39,25,rf'\(Z^{{({k+1})}}\)',size=18)
    edge('projection-z1',(444,317),(468,317))
    edge('projection-z2',(421,341),(570,336),[(421,351),(570,351)])
    edge('contrastive-arrow',(510,317),(550,317),color='#A64D79',dashed=True)
    edge('contrastive-arrow-reverse',(550,322),(510,322),color='#A64D79',dashed=True)
    text('original-graph-encoding',31,362,570,29,
         r'After pretraining: \(\bar H=\operatorname{RowNorm}(g_\psi(X,A))\)',size=19)

    text('initial-title',651,84,124,45,'Initial region',size=20,bold=True)
    text('split-title',785,84,144,45,'2-means split',size=20,bold=True)
    text('further-title',940,84,146,45,'Further splits',size=20,bold=True)
    text('partition-title',1110,84,261,45,'Final partition',size=20,bold=True)
    init_ids=[];split_ids=[];further_ids=[]
    for id,c in old_cells.items():
        g=geometries(c)
        if not g or 'ellipse;' not in c.get('style',''):continue
        x,y,w,h=g
        if 840<=x+w/2<=1040 and 130<=y+h/2<=339:init_ids.append(id)
        if 1047<=x+w/2<=1145 and 125<=y+h/2<=312:split_ids.append(id)
        if 1159<=x+w/2<=1311 and 126<=y+h/2<=305:further_ids.append(id)
    clone_group(init_ids,'init',(845,130,195,205),(654,165,125,158))
    clone_group(split_ids,'split',(1050,125,96,183),(792,155,128,178))
    clone_group(further_ids,'further',(1164,129,142,166),(945,159,143,167))
    partition_ids=[]
    for i in range(359,564):
        for prefix in ['e','r']:
            id=f'{prefix}{i}'
            if id in old_cells and ('ellipse;' in old_cells[id].get('style','') or old_cells[id].get('value')=='\u00d7'):
                partition_ids.append(id)
    clone_group(partition_ids,'partition',(1337,108,260,213),(1118,132,257,230),font_size=18)
    edge('h-to-partition',(615,231),(635,231))
    edge('initial-to-split',(779,243),(792,243))
    edge('split-to-further',(924,243),(943,243))
    edge('further-to-final',(1091,243),(1114,243))
    text('partition-note',650,346,456,26,r'Label-free splits; \(K\le B_{\max}=64\)',size=19,color=MUTED)
    text('partition-projection-note',650,374,490,26,'Disjoint membership; circles show 2-D covers.',size=17,color=MUTED)

    text('selected-title',32,498,218,32,'(a) Selected ball',size=21,bold=True)
    text('votes-title',253,498,223,32,'(b) Label proportions',size=21,bold=True)
    text('target-title',496,498,324,32,'(c) Soft-target construction',size=21,bold=True)
    selected=[f'e{i}' for i in range(568,591)]+['e_v10_target_ring','r_v10_target_v']
    clone_group(selected,'selected',(52.26,540,244,244),(42,548,205,205),font_size=19)
    text('selected-ball-label',72,762,153,25,r'Granular ball \(B_j\)',size=20)
    text('exclude-note',35,800,218,26,r"Exclude \(v\)'s vote",size=19,color=MUTED)
    edge('selected-to-votes',(250,648),(281,648))
    p=[1/12,1/12,8/12,1/12,1/12]
    alpha=(max(p)-.2)/.8
    q=[.5*z+(.5 if i==2 else 0) for i,z in enumerate(p)]
    own=[0,1-alpha,0,0,0]
    weighted=[alpha*z for z in q]
    target=[a+b for a,b in zip(own,weighted)]
    plotbars('vote-chart',292,583,152,116,p,title=r'\(p_{j,-v}\)',axis_one=True)
    edge('votes-to-target',(453,648),(482,648))
    for prefix,x,values,label,formula in [
        ('own',493,own,'Own label',r'\(\times(1-\alpha_{j,-v})\)'),
        ('consensus',607,weighted,'Consensus',r'\(\times\alpha_{j,-v}\)'),
        ('target',722,target,'Soft target',r'\(t_v\)')]:
        text(prefix+'-heading',x-9,546,113,27,label,size=19)
        text(prefix+'-formula',x-13,576,113,30,formula,size=18)
        plotbars(prefix+'-chart',x,620,84,81,values,axis_one=prefix=='own')
    text('sum-plus',578,650,29,29,'+',size=26,bold=True)
    text('sum-equals',693,650,29,29,'=',size=25,bold=True)
    text('sharpen-formula',258,757,277,47,
         r'\(q_{j,-v}=(1-\beta)p_{j,-v}+\beta e_{m_{j,-v}}\)',size=18)
    text('tie-note',259,814,263,46,'Ties retain the<br>vote proportions.',size=18,color=MUTED)
    text('weight-note',531,754,296,48,'Stronger label agreement gives<br>more weight to consensus.',size=19,color=MUTED)
    text('final-formula',527,812,304,48,
         r'\(t_v=(1-\alpha_{j,-v})e_{\tilde y_v}+\alpha_{j,-v}q_{j,-v}\)',size=20)

    selected_source=next(c for c in cells if c.get('id')=='partition-e563')
    sx,sy,sw,sh=geometries(selected_source)
    edge('selected-ball-link',(sx+sw/2,sy+sh),(42,650),
         [(sx+sw/2,425),(3,425),(3,650)],color=ORANGE,lw=1.6)

    text('training-title',897,494,245,29,'MLP training',size=22,bold=True,align='left')
    text('training-mlp-title',988,544,132,28,r'MLP \(f_\theta\)',size=21,bold=True)
    feature('train-feature',902,605)
    text('train-feature-label',895,635,78,29,r'\(x_v\)',size=21)
    mlp('train-mlp',1002,580,76,66)
    edge('train-input',(966,613),(988,613))
    edge('train-output',(1090,613),(1118,613))
    text('prediction-label',1112,550,103,28,r'\(\hat p_v\)',size=22)
    plotbars('prediction-chart',1130,588,80,69,[.10,.24,.55,.07,.04])
    box('training-loss',1250,583,119,72,'Soft cross-<br>entropy',fill='#F8FAFB',stroke=INK,size=20,bold=True)
    edge('prediction-to-loss',(1215,613),(1248,613))
    edge('target-to-loss',(811,618),(1310,580),[(861,618),(861,533),(1310,533)])
    text('target-link-label',1154,505,217,25,r'Soft target \(t_v\)',size=19,color=MUTED)
    text('training-scope',898,686,460,28,r'Training nodes \(v\in\mathcal V_L\)',size=19,color=MUTED,align='left')
    edge('phase-separator',(895,717),(1370,717),color='#CAD2D9',lw=.9,dashed=True,arrow=False)
    text('inference-title',897,750,430,28,'Graph-free inference',size=22,bold=True,align='left')
    feature('infer-feature',902,821)
    text('infer-feature-label',895,846,78,29,r'\(x_u\)',size=21)
    mlp('infer-mlp',1002,790,76,66)
    text('same-mlp-label',984,869,113,24,r'Same \(f_\theta\)',size=18,color=MUTED)
    edge('infer-input',(966,829),(988,829))
    edge('infer-output',(1090,829),(1118,829))
    text('inference-pred-label',1112,757,103,29,r'\(\hat p_u\)',size=22)
    plotbars('infer-chart',1130,801,80,59,[.1,.24,.55,.07,.04])
    for i in [0,1,3,4]:
        c=next(c for c in cells if c.get('id')==f'infer-chart-bar{i}')
        c.set('style',f'rounded=0;fillColor=#E7EDF2;strokeColor=#B6C2CC;strokeWidth=.65;')
    box('class-output',1250,803,119,55,'Class 3',fill='#F8FAFB',stroke=INK,size=21,bold=True)
    edge('class-prediction',(1215,829),(1248,829))

    edge('legend-divider',(13,925),(1388,925),color='#B7C4CE',lw=1,arrow=False)
    text('legend-head',20,938,235,26,'Observed training labels:',size=18,bold=True,align='left')
    for i,color in enumerate(COLORS):
        circle(f'legend-class-{i}',260+i*69,942,16,fill=color,stroke='#4B5862',lw=.7)
        text(f'legend-class-text-{i}',279+i*69,937,44,26,f'C{i+1}',size=18,align='left')
    circle('legend-hollow',620,942,16,fill='#F1F3F5',stroke='#9AA7B2',lw=1)
    text('legend-hollow-text',643,937,182,26,'Non-voting node',size=18,align='left')
    text('deployment-note',878,937,498,26,'Inference uses only node attributes.',size=19,bold=True,align='left')
    assert abs(sum(p)-1)<1e-12 and abs(sum(target)-1)<1e-12
    return dict(votes=[1,1,8,1,1],alpha=alpha,local_consensus=q,soft_target=target,
                old_selected_node_count=22,new_selected_node_count=22,
                imported_yaml_error='Too many nodes (1272): maximum is 100',
                workflow='Native mxGraph edit after unusable YAML import; draw.io Desktop export')


def main():
    OUT.mkdir(exist_ok=True,parents=True);WORK.mkdir(exist_ok=True,parents=True)
    original=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    checks=create()
    path=OUT/'fig1_gblcss_overview_drawio_v21_readable.drawio'
    E.ElementTree(doc).write(path,encoding='utf-8',xml_declaration=True)
    E.parse(path)
    assert len(used)==len(set(used))
    for c in cells:
        for attr in ['source','target']:
            if c.get(attr):assert c.get(attr) in used
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==original
    checks.update(source_unchanged=True,original_sha256=original,object_count=len(used))
    (WORK/'fig1_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(path)


if __name__=='__main__':
    main()
