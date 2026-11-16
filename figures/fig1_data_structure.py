"""Figure 1 (methods and appraisal versions): many individuals, few groups.

Writes output/figure1_methods.svg (generic labels first) and output/figure1_appraisal.svg
(psychedelic labels first). Matrix values and report excerpts are illustrative.
"""
import os, random, math
from svg import OUT
random.seed(7)
W,H=1400,670
C={'LSD':'#0072B2','Psilocybin':'#E69F00','MDMA':'#CC79A7','DMT':'#009E73'}
drugs=['LSD','Psilocybin','MDMA','DMT']; n={'LSD':4,'Psilocybin':2,'MDMA':2,'DMT':1}
quotes={'LSD':['“…the walls began breathing…”','“…colours had sounds…”','“…time folded in…”','“…I laughed for an hour…”'],
 'Psilocybin':['“…one with the trees…”','“…nausea, then calm…”'],
 'MDMA':['“…I loved everyone there…”','“…jaw clenching all night…”'],
 'DMT':['“…entities greeted me…”']}
rows=[(d,q) for d in drugs for q in quotes[d]]
P,Q=10,8
# drug-level means and profiles
mu={d:[random.uniform(.2,.8) for _ in range(P)] for d in drugs}
prof={d:[random.choice([0,0,.15,.4,.7,1.0]) for _ in range(Q)] for d in drugs}
prof['MDMA']=[0,0,0,.35,0,0,.6,0]
xs=[[min(1,max(0,mu[d][j]+random.gauss(0,.18))) for j in range(P)] for d,_ in rows]
xbar={d:[sum(xs[i][j] for i,(dd,_) in enumerate(rows) if dd==d)/n[d] for j in range(P)] for d in drugs}
o=[]
a=o.append
def t(x,y,s,size=13,fill='#1b2128',anchor='start',weight='400',style='normal',fam='Helvetica, Arial, sans-serif',extra=''):
    a(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" font-style="{style}" font-family="{fam}" {extra}>{s}</text>')
def vdots(x,y):
    for k in range(3): a(f'<circle cx="{x}" cy="{y-14+k*7}" r="1.8" fill="#8a96a3"/>')
def ytilde(size,w='700'):
    """Italic Y with a tilde (Ỹ): Georgia has no precomposed Ỹ, so a spacing tilde is placed over the Y."""
    adv=0.73 if w=='700' else 0.615       # advance of italic Y in Georgia (bold, regular), em
    return f'Y<tspan dx="{-(adv*0.5+0.25+0.04-0.06)*size:.1f}" dy="{-0.24*size:.1f}">˜</tspan>'
def xbarlab(x,y,size=16,w='700'):
    t(x,y,'X',size,weight=w,style='italic',fam='Georgia, serif')
    a(f'<line x1="{x+2}" y1="{y-size*0.82}" x2="{x+size*0.72}" y2="{y-size*0.82}" stroke="#1b2128" stroke-width="1.4"/>')
def grey(v):
    g=int(245-v*190); return f'rgb({g},{g},{g+4 if g<251 else g})'
import json
T_ALL={'methods': {'titleA': 'The data: many individuals, few groups', 'rep': ('Individuals', 'experience reports'), 'X': ('features, p = 100', 'report embeddings'), 'within': 'and within groups', 'Y': ('annotations, q = 19', 'receptor affinities, inherited'), 'const': 'constant within groups', 'bracket': 'analysed at individual level: N = 4,496 rows', 'Yt': ('group annotations, M = 21', 'drug receptor profiles'), 'every': 'every member', 'meas': ('measured once per group', '(here: binding assays, Ki)'), 'info': 'information about annotations:', 'xb': ('group means', 'drug means of X'), 'D': ('group sizes', 'reports per drug'), 'yb': ('group annotations', 'drug profiles'), 'rowh': 'row height proportional to group size', 'ident': ['every cross-product is a weighted sum over', 'the M group means; variation within', 'groups drops out'], 'dots': '4 individuals → 1 group mean', 'bars': ['individuals (reports)', 'groups (drugs)', 'effective groups']}, 'appraisal': {'titleA': 'The data: many reports, few drugs', 'rep': ('Experience reports', 'individuals'), 'X': ('report embeddings, p = 100', 'individual features'), 'within': 'and within drugs', 'Y': ('receptor affinities, q = 19', 'annotations, inherited'), 'const': 'constant within drugs', 'bracket': 'analysed at report level: N = 4,496 rows', 'Yt': ('drug receptor profiles, M = 21', 'group annotations'), 'every': 'every report', 'meas': ('affinities (Ki) measured', 'in binding assays, once per drug'), 'info': 'information about receptors:', 'xb': ('drug means of X', 'group means'), 'D': ('reports per drug', 'group sizes'), 'yb': ('drug profiles', 'group annotations'), 'rowh': 'row height proportional to number of reports', 'ident': ['every cross-product is a weighted sum over', 'the M drug means; variation between reports', 'of the same drug drops out'], 'dots': '4 LSD reports → 1 drug mean', 'bars': ['reports (individuals)', 'drugs (groups)', 'effective drugs']}}
for MODE,T in T_ALL.items():
    H=670 if MODE=='appraisal' else 640   # methods version has no sample-size strip at the bottom
    o.clear()
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
    a('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#4d5966"/></marker></defs>')
    a(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    # panel a
    t(20,40,'a',20,weight='700'); t(46,40,T['titleA'],17,weight='700')
    y0,rh,gap=120,34,6
    ry=[y0+i*(rh+gap) for i in range(len(rows))]
    t(20,100,T['rep'][0],13,weight='700'); t(20,114,T['rep'][1],11,fill='#8a96a3')
    for i,(d,q) in enumerate(rows):
        y=ry[i]
        a(f'<rect x="20" y="{y}" width="236" height="{rh}" rx="4" fill="#f4f6f8" stroke="#d7dde3"/>')
        a(f'<rect x="26" y="{y+8}" width="66" height="18" rx="9" fill="{C[d]}"/>')
        t(59,y+21,d,10.5,fill='#ffffff',anchor='middle',weight='700')
        t(100,y+21,q,11.5,fill='#33404d',style='italic')
    yend=ry[-1]+rh
    vdots(138,yend+26); t(138,yend+46,'9 of N = 4,496 shown',12,fill='#4d5966',anchor='middle')
    # arrow embed
    my=(ry[0]+yend)/2
    a(f'<line x1="262" y1="{my}" x2="306" y2="{my}" stroke="#4d5966" stroke-width="1.6" marker-end="url(#ah)"/>')
    t(284,my-8,'embed',11,fill='#4d5966',anchor='middle')
    # X matrix
    xx=322; cw=15
    t(xx,82,'X',16,weight='700',style='italic',fam='Georgia, serif'); t(xx+16,82,'(N × p)',12,fill='#4d5966')
    t(xx,98,T['X'][0],11.5,fill='#4d5966'); t(xx,112,T['X'][1],11,fill='#8a96a3')
    for i,(d,_) in enumerate(rows):
        y=ry[i]
        a(f'<rect x="{xx-8}" y="{y}" width="5" height="{rh}" fill="{C[d]}"/>')
        for j in range(P):
            a(f'<rect x="{xx+j*cw}" y="{y}" width="{cw}" height="{rh}" fill="{grey(xs[i][j])}" stroke="#ffffff" stroke-width="1"/>')
    vdots(xx+P*cw/2,yend+26)
    t(xx+P*cw/2,yend+46,'varies between',11.5,fill='#4d5966',anchor='middle')
    t(xx+P*cw/2,yend+61,T['within'],11.5,fill='#4d5966',anchor='middle')
    t(xx+P*cw/2,yend+78,'10 of p = 100 columns shown',10.5,fill='#8a96a3',anchor='middle')
    # Y matrix
    yx=520
    t(yx,82,'Y',16,weight='700',style='italic',fam='Georgia, serif'); t(yx+16,82,'(N × q)',12,fill='#4d5966')
    t(yx,98,T['Y'][0],11.5,fill='#4d5966'); t(yx,112,T['Y'][1],11,fill='#8a96a3')
    blocks={}
    for i,(d,_) in enumerate(rows):
        y=ry[i]; blocks.setdefault(d,[]).append(y)
        for j in range(Q):
            v=prof[d][j]
            a(f'<rect x="{yx+j*cw}" y="{y}" width="{cw}" height="{rh}" fill="{C[d]}" fill-opacity="{0.06+0.9*v:.2f}" stroke="#ffffff" stroke-width="1"/>')
    vdots(yx+Q*cw/2,yend+26)
    t(yx+Q*cw/2,yend+46,T['const'],11.5,fill='#4d5966',anchor='middle')
    t(yx+Q*cw/2,yend+78,'8 of q = 19 columns shown',10.5,fill='#8a96a3',anchor='middle')
    bx=yx+Q*cw+8
    centers={}
    for d in drugs:
        ys=blocks[d]; top=ys[0]; bot=ys[-1]+rh; centers[d]=(top+bot)/2
        a(f'<path d="M{bx},{top+2} h6 V{bot-2} h-6" fill="none" stroke="{C[d]}" stroke-width="1.6"/>')
        t(bx+10,(top+bot)/2+4,f'×{n[d]}',12,fill=C[d],weight='700')
    # report-level bracket over X and Y
    a(f'<path d="M{xx-8},60 v-6 H{yx+Q*cw} v6" fill="none" stroke="#8a96a3" stroke-width="1.2" stroke-dasharray="4 3"/>')
    t(yx+Q*cw,47,T['bracket'],11.5,fill='#4d5966',anchor='end')
    # Ytilde
    tx=760; th=26
    t(tx,82,ytilde(16),16,weight='700',style='italic',fam='Georgia, serif'); t(tx+16,82,'(M × q)',12,fill='#4d5966')
    t(tx,98,T['Yt'][0],11.5,fill='#4d5966'); t(tx,112,T['Yt'][1],11,fill='#8a96a3')
    for d in drugs:
        y=centers[d]-th/2
        for j in range(Q):
            v=prof[d][j]
            a(f'<rect x="{tx+j*cw}" y="{y}" width="{cw}" height="{th}" fill="{C[d]}" fill-opacity="{0.06+0.9*v:.2f}" stroke="#ffffff" stroke-width="1"/>')
        a(f'<rect x="{tx}" y="{y}" width="{Q*cw}" height="{th}" fill="none" stroke="{C[d]}" stroke-width="1.4"/>')
        t(tx+Q*cw+8,centers[d]+4,d,12,fill=C[d],weight='700')
        a(f'<line x1="{tx-4}" y1="{centers[d]}" x2="{bx+34}" y2="{centers[d]}" stroke="{C[d]}" stroke-width="1.4" marker-end="url(#ah)"/>')
    t((tx+bx+30)/2-8,ry[0]-12,'inherited by',11,fill='#4d5966',anchor='middle')
    t((tx+bx+30)/2-8,ry[0]-0,T['every'],11,fill='#4d5966',anchor='middle')
    vdots(tx+Q*cw/2,yend+26)
    t(tx+Q*cw/2+10,yend+46,T['meas'][0],11.5,fill='#4d5966',anchor='middle')
    t(tx+Q*cw/2+10,yend+61,T['meas'][1],11.5,fill='#4d5966',anchor='middle')
    t(tx+Q*cw/2+10,yend+78,'4 of M = 21 drugs shown',10.5,fill='#8a96a3',anchor='middle')
    a(f'<path d="M{tx},{yend+90} v6 H{tx+Q*cw+60} v-6" fill="none" stroke="{"#1d5d7f"}" stroke-width="1.4"/>')
    t(tx+(Q*cw+60)/2,yend+112,T['info'],11.5,fill='#1d5d7f',anchor='middle',weight='700')
    t(tx+(Q*cw+60)/2,yend+127,'M = 21 rows',11.5,fill='#1d5d7f',anchor='middle',weight='700')
    # divider
    a(f'<line x1="965" y1="30" x2="965" y2="{H-30}" stroke="#d7dde3" stroke-width="1"/>')
    # panel b
    bx0=990
    t(bx0,40,'b',20,weight='700'); t(bx0+26,40,'What every cross-product sees',17,weight='700')
    xbx=bx0+10; unit=24; yb=126
    xbarlab(xbx,82); t(xbx+16,82,'(M × p)',12,fill='#4d5966')
    t(xbx,98,T['xb'][0],11.5,fill='#4d5966'); t(xbx,112,T['xb'][1],11,fill='#8a96a3')
    dlx=xbx+P*cw+14
    t(dlx+16,82,'D',16,weight='700',style='italic',fam='Georgia, serif'); t(dlx,98,T['D'][0],11.5,fill='#4d5966'); t(dlx,112,T['D'][1],11,fill='#8a96a3')
    tbx=dlx+92
    t(tbx,82,ytilde(16),16,weight='700',style='italic',fam='Georgia, serif'); t(tbx+16,82,'(M × q)',12,fill='#4d5966')
    t(tbx,98,T['yb'][0],11.5,fill='#4d5966'); t(tbx,112,T['yb'][1],11,fill='#8a96a3')
    y=yb
    for d in drugs:
        h=unit*n[d]
        a(f'<rect x="{xbx-8}" y="{y}" width="5" height="{h}" fill="{C[d]}"/>')
        for j in range(P):
            a(f'<rect x="{xbx+j*cw}" y="{y}" width="{cw}" height="{h}" fill="{grey(xbar[d][j])}" stroke="#ffffff" stroke-width="1"/>')
        a(f'<rect x="{dlx}" y="{y+1}" width="{8*n[d]}" height="{h-2}" fill="{C[d]}" fill-opacity="0.35"/>')
        t(dlx+8*n[d]+5,y+h/2+4,f'{n[d]}',11.5,fill='#4d5966')
        for j in range(Q):
            v=prof[d][j]
            a(f'<rect x="{tbx+j*cw}" y="{y}" width="{cw}" height="{h}" fill="{C[d]}" fill-opacity="{0.06+0.9*v:.2f}" stroke="#ffffff" stroke-width="1"/>')
        y+=h+6
    yb2=y
    t(bx0+10,yb2+18,T['rowh'],11,fill='#8a96a3')
    # identity
    fy=yb2+56
    fx=bx0+10
    a(f'<text x="{fx}" y="{fy}" font-family="Georgia, serif" font-size="20" fill="#1b2128" font-style="italic">X<tspan font-size="12" dy="-9">T</tspan><tspan dy="9">Y  =  </tspan></text>')
    xbarlab(fx+70,fy,20,'400')
    a(f'<text x="{fx+86}" y="{fy}" font-family="Georgia, serif" font-size="20" fill="#1b2128" font-style="italic"><tspan font-size="12" dy="-9">T</tspan><tspan dy="9"> D {ytilde(20,"400")}</tspan></text>')
    for k,ln in enumerate(T['ident']): t(bx0+10,fy+22+16*k,ln,12,fill='#33404d')
    # dot strip: reports -> mean
    sy=fy+96
    a(f'<line x1="{bx0+10}" y1="{sy}" x2="{bx0+250}" y2="{sy}" stroke="#d7dde3"/>')
    for v in [.28,.42,.55,.7]:
        a(f'<circle cx="{bx0+10+v*240}" cy="{sy}" r="5" fill="{C["LSD"]}" fill-opacity="0.3"/>')
    a(f'<rect x="{bx0+10+.49*240-1.5}" y="{sy-11}" width="3" height="22" fill="{C["LSD"]}"/>')
    t(bx0+10,sy-16,T['dots'],11,fill='#4d5966')
    # sample-size strip (log scale); appraisal version only, the methods paper drops it
    ly=sy+44
    items=[(T['bars'][0],4496,'N = 4,496','#8a96a3'),(T['bars'][1],21,'M = 21','#1d5d7f'),(T['bars'][2],10.8,'M_eff ≤ 10.8 (Kish, size weights)','#1d5d7f')]
    if MODE=='appraisal':
        t(bx0+10,ly,'How many rows count',12.5,weight='700'); t(bx0+150,ly,'(log scale)',11,fill='#8a96a3')
    for k,(lab,v,txt,col) in enumerate(items if MODE=='appraisal' else []):
        yy=ly+14+k*22
        L=50*math.log10(v)
        t(bx0+10,yy+11,lab,11,fill='#4d5966')
        a(f'<rect x="{bx0+150}" y="{yy}" width="{L:.0f}" height="14" fill="{col}" fill-opacity="{0.85 if k==1 else 0.5}"/>')
        t(bx0+156+L,yy+11,txt,11,fill=('#a06a00' if 'fill' in txt else '#33404d'))
    a('</svg>')
    
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f'figure1_{MODE}.svg'), 'w', encoding='utf-8').write('\n'.join(o))
