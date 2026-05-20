# Phase 1 — Fitness-AQA Dataset EDA
#
# Companion notebook to `.planning/phases/01-dataset-consolidation-eda/01-DATASET-REPORT.md`.
# Run in Colab. Mounts the user's Drive shortcut `My Drive/Fitness-AQA_dataset_release`.
#
# Jupytext "percent" format: each `# %%` marker starts a new cell.
# Open in Colab via `pip install jupytext && jupytext --to ipynb 01_dataset_eda.py`,
# or paste cells one at a time.

# %% [markdown]
# ## Step 0 — verify Colab can reach the dataset in Drive

# %%
import os, zipfile, time
from google.colab import drive
drive.mount('/content/drive')

MYDRIVE = '/content/drive/MyDrive'
ROOT = os.path.join(MYDRIVE, 'Fitness-AQA_dataset_release')

print('ROOT :', ROOT)
print('exists:', os.path.exists(ROOT))
if not os.path.exists(ROOT):
    print('\n!! Not found. Top level of My Drive:')
    for n in sorted(os.listdir(MYDRIVE))[:50]:
        print('  ', n)
    raise SystemExit('Fix the ROOT path above, then re-run.')

print('\n=== TREE (folders, json, zip — with sizes) ===')
zips = []
for dp, dns, fns in os.walk(ROOT):
    dns.sort()
    depth = dp[len(ROOT):].count(os.sep)
    print('  ' * depth + os.path.basename(dp) + '/')
    for f in sorted(fns):
        full = os.path.join(dp, f)
        try:
            print('  ' * (depth + 1) + f'{f}  ({os.path.getsize(full)/1e6:.2f} MB)')
        except Exception as e:
            print('  ' * (depth + 1) + f'{f}  (SIZE ERROR: {e})')
        if f.lower().endswith('.zip'):
            zips.append(full)

print(f'\n=== ZIP READ TEST — {len(zips)} zips ===')
ok = 0
for z in zips:
    rel = os.path.relpath(z, ROOT)
    try:
        t0 = time.time()
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            member = next((n for n in names if not n.endswith('/')), None)
            data = zf.read(member) if member else b''
        print(f'  OK    {rel} — {len(names)} entries, read "{os.path.basename(member)}" '
              f'{len(data)/1e6:.2f} MB ({time.time()-t0:.1f}s)')
        ok += 1
    except Exception as e:
        print(f'  FAIL  {rel} — {type(e).__name__}: {e}')

print(f'\nVERDICT: {ok}/{len(zips)} zips fully readable.')

# %% [markdown]
# ## Cell 1 — dataset audit & archive integrity (DATA-01)
#
# Confirms member counts against the verified inventory and quantifies the
# Squat 1,739-vs-1,623 anomaly by ID-diffing videos.zip against the label keys.

# %%
import os, zipfile, json
ROOT = '/content/drive/MyDrive/Fitness-AQA_dataset_release'

EXPECT = {
    'OHP/Labeled_Dataset/videos.zip': 2367,
    'OHP/Unlabeled_Dataset/videos.zip': 5490,
    'OHP/Unlabeled_Dataset/bar_trajectories_raw.zip': 5490,
    'Squat/Labeled_Dataset/videos.zip': 1739,
    'Squat/Unlabeled_Dataset/videos.zip': 4970,
    'Squat/Unlabeled_Dataset/bar_trajectories_raw.zip': 4970,
    'Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/images.zip': 3738,
    'BarbellRow/Labeled_Dataset/barbellrow_images_raw.zip': 33759,
}

print('=== ARCHIVE INTEGRITY ===')
members = {}
for rel, exp in EXPECT.items():
    with zipfile.ZipFile(os.path.join(ROOT, rel)) as zf:
        names = [n for n in zf.namelist() if not n.endswith('/')]
    members[rel] = names
    flag = 'OK' if len(names) == exp else f'!! MISMATCH (expected {exp})'
    print(f'  {len(names):>6}  {rel}   {flag}')

print('\n=== SQUAT LABELED: videos vs labels ===')
vid_ids = {os.path.splitext(os.path.basename(n))[0]
           for n in members['Squat/Labeled_Dataset/videos.zip']}
L = 'Squat/Labeled_Dataset/Labels'
kie = json.load(open(os.path.join(ROOT, L, 'error_knees_inward.json')))
kfe = json.load(open(os.path.join(ROOT, L, 'error_knees_forward.json')))
label_ids = set(kie) | set(kfe)
print(f'  videos.zip unique IDs  : {len(vid_ids)}')
print(f'  KIE label keys         : {len(kie)}')
print(f'  KFE label keys         : {len(kfe)}')
print(f'  union of label keys    : {len(label_ids)}')
print(f'  videos WITHOUT a label : {len(vid_ids - label_ids)}')
print(f'  labels WITHOUT a video : {len(label_ids - vid_ids)}')
print(f'  sample unlabeled IDs   : {sorted(vid_ids - label_ids)[:12]}')
print(f'  KIE and KFE same keys? : {set(kie) == set(kfe)}')

# %% [markdown]
# ## Cell 2 — label & split count reconciliation (DATA-02)
#
# Per-error totals, class balance, train/val/test sizes, and disjoint/coverage
# checks for all 7 errors. Reconciles against published Fitness-AQA figures.

# %%
import os, json
ROOT = '/content/drive/MyDrive/Fitness-AQA_dataset_release'
def jload(*p): return json.load(open(os.path.join(ROOT, *p)))

SPEC = {
 'Squat KIE':    ('Squat/Labeled_Dataset/Labels/error_knees_inward.json','interval',
   'Squat/Labeled_Dataset/Splits',('train_keys','val_keys','test_keys'),1623,14.29),
 'Squat KFE':    ('Squat/Labeled_Dataset/Labels/error_knees_forward.json','interval',
   'Squat/Labeled_Dataset/Splits',('train_keys','val_keys','test_keys'),1623,68.33),
 'Squat Shallow':('Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/labels_shallow_depth.json','binary',
   'Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/splits',('train_ids','val_ids','test_ids'),3611,43.87),
 'OHP Elbows':   ('OHP/Labeled_Dataset/Labels/error_elbows.json','interval',
   'OHP/Labeled_Dataset/Splits',('train_keys','val_keys','test_keys'),2260,34.38),
 'OHP Knees':    ('OHP/Labeled_Dataset/Labels/error_knees.json','interval',
   'OHP/Labeled_Dataset/Splits',('train_keys','val_keys','test_keys'),2260,25.49),
 'BR Lumbar':    ('BarbellRow/Labeled_Dataset/Labels/labels_lumbar_error.json','binary',
   'BarbellRow/Labeled_Dataset/Splits/Splits_Lumbar_Error',('train_ids','val_ids','test_ids'),14778,15.68),
 'BR Torso':     ('BarbellRow/Labeled_Dataset/Labels/labels_torso_angle_error.json','binary',
   'BarbellRow/Labeled_Dataset/Splits/Splits_TorsoAngle_Error',('train_ids','val_ids','test_ids'),17030,9.21),
}

hdr = f'{"error":15}{"labelfile":>10}{"splitsum":>9}{"pub":>8}{"%err":>9}{"%pub":>8}   tr/va/te  ov  miss'
print(hdr); print('-'*len(hdr))
for name,(lp,kind,sf,sfiles,pubtot,pubpct) in SPEC.items():
    d = jload(lp)
    pos = (sum(1 for v in d.values() if v) if kind=='interval'
           else sum(1 for v in d.values() if int(v)==1))
    pct = 100*pos/len(d)
    sets = [set(jload(sf,f+'.json')) for f in sfiles]
    sz = [len(s) for s in sets]
    ssum = sum(sz)
    ov = len(sets[0]&sets[1])+len(sets[0]&sets[2])+len(sets[1]&sets[2])
    miss = len((sets[0]|sets[1]|sets[2]) - set(d))
    f1 = ' L!' if len(d)!=pubtot else ''
    f2 = ' S!' if ssum!=pubtot else ''
    print(f'{name:15}{len(d):>10}{ssum:>9}{pubtot:>8}{pct:>8.2f}%{pubpct:>7.2f}%   '
          f'{sz[0]}/{sz[1]}/{sz[2]}  {ov}  {miss}{f1}{f2}')

# %% [markdown]
# ## Cell 3 — clip properties (DATA-03)
#
# Decodes 50 random labeled clips per video exercise to characterize fps, frame
# count, duration, and resolution distributions.

# %%
import os, zipfile, random, tempfile, collections, cv2, numpy as np
ROOT = '/content/drive/MyDrive/Fitness-AQA_dataset_release'

def probe_sample(zip_rel, n=50, seed=0):
    rows = []
    with zipfile.ZipFile(os.path.join(ROOT, zip_rel)) as zf:
        names = [x for x in zf.namelist() if x.lower().endswith('.mp4')]
        random.Random(seed).shuffle(names)
        with tempfile.TemporaryDirectory() as tmp:
            for nm in names[:n]:
                p = os.path.join(tmp, os.path.basename(nm))
                with open(p, 'wb') as fh:
                    fh.write(zf.read(nm))
                cap = cv2.VideoCapture(p)
                fps = cap.get(cv2.CAP_PROP_FPS)
                fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                rows.append((os.path.basename(nm), w, h, fps, fc, fc/fps if fps else 0))
    return rows

for label, zrel in [('Squat labeled', 'Squat/Labeled_Dataset/videos.zip'),
                     ('OHP labeled',   'OHP/Labeled_Dataset/videos.zip')]:
    rows = probe_sample(zrel, n=50)
    fps = [r[3] for r in rows]; fc = [r[4] for r in rows]; dur = [r[5] for r in rows]
    res = collections.Counter((r[1], r[2]) for r in rows)
    print(f'\n=== {label}  (n={len(rows)} sampled) ===')
    print(f'  fps      : min {min(fps):.1f}  max {max(fps):.1f}  mean {np.mean(fps):.2f}')
    print(f'  frames   : min {min(fc)}  max {max(fc)}  mean {np.mean(fc):.1f}  median {int(np.median(fc))}')
    print(f'  duration : min {min(dur):.2f}s  max {max(dur):.2f}s  mean {np.mean(dur):.2f}s')
    print(f'  resolution WxH counts: {dict(res.most_common())}')

# %% [markdown]
# ## Cell 4 — class balance & error co-occurrence (DATA-03)

# %%
import os, json
import matplotlib.pyplot as plt
ROOT = '/content/drive/MyDrive/Fitness-AQA_dataset_release'
def jload(*p): return json.load(open(os.path.join(ROOT, *p)))

def pos_set(path, kind):
    d = jload(path)
    if kind == 'interval':
        return {k for k, v in d.items() if v}, set(d)
    return {k for k, v in d.items() if int(v) == 1}, set(d)

ERR = [
 ('Squat KIE','Squat/Labeled_Dataset/Labels/error_knees_inward.json','interval',None),
 ('Squat KFE','Squat/Labeled_Dataset/Labels/error_knees_forward.json','interval',None),
 ('Squat Shallow','Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/labels_shallow_depth.json','binary',None),
 ('OHP Elbows','OHP/Labeled_Dataset/Labels/error_elbows.json','interval',None),
 ('OHP Knees','OHP/Labeled_Dataset/Labels/error_knees.json','interval',None),
 ('BR Lumbar','BarbellRow/Labeled_Dataset/Labels/labels_lumbar_error.json','binary',
   ('BarbellRow/Labeled_Dataset/Splits/Splits_Lumbar_Error',('train_ids','val_ids','test_ids'))),
 ('BR Torso','BarbellRow/Labeled_Dataset/Labels/labels_torso_angle_error.json','binary',
   ('BarbellRow/Labeled_Dataset/Splits/Splits_TorsoAngle_Error',('train_ids','val_ids','test_ids'))),
]
names, pcts, totals = [], [], []
for nm, path, kind, splitspec in ERR:
    pos, allids = pos_set(path, kind)
    if splitspec:
        sf, files = splitspec
        official = set()
        for f in files: official |= set(jload(sf, f + '.json'))
        pos, allids = pos & official, official
    p = 100 * len(pos) / len(allids)
    names.append(nm); pcts.append(p); totals.append(len(allids))
    print(f'  {nm:15} {len(allids):>6} samples  {len(pos):>6} erroneous  {p:6.2f}%')

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(names, pcts, color='#4a90d9')
ax.set_ylabel('% erroneous'); ax.set_ylim(0, 100)
ax.set_title('Fitness-AQA — class balance per error (official benchmark set)')
for b, p, t in zip(bars, pcts, totals):
    ax.text(b.get_x() + b.get_width()/2, p + 2, f'{p:.1f}%\nn={t}', ha='center', fontsize=8)
plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.show()

def cooc(title, pa, pb, all_ids, la, lb):
    both = len(pa & pb); a_only = len(pa - pb); b_only = len(pb - pa)
    neither = len(all_ids) - both - a_only - b_only
    print(f'\n{title}  (n={len(all_ids)})')
    print(f'            {lb}+     {lb}-')
    print(f'  {la}+  {both:>7} {a_only:>7}')
    print(f'  {la}-  {b_only:>7} {neither:>7}')

kie_p, kie_all = pos_set('Squat/Labeled_Dataset/Labels/error_knees_inward.json', 'interval')
kfe_p, _       = pos_set('Squat/Labeled_Dataset/Labels/error_knees_forward.json', 'interval')
cooc('Squat  KIE x KFE', kie_p, kfe_p, kie_all, 'KIE', 'KFE')
elb_p, ohp_all = pos_set('OHP/Labeled_Dataset/Labels/error_elbows.json', 'interval')
kne_p, _       = pos_set('OHP/Labeled_Dataset/Labels/error_knees.json', 'interval')
cooc('OHP  Elbows x Knees', elb_p, kne_p, ohp_all, 'Elb', 'Kne')

# %% [markdown]
# ## Cell 5 — barbell trajectories (the SSL domain-knowledge signal)
#
# Squat ships processed y-centers — clean parabolic curves, ready for MD SSL.
# OHP ships raw YOLO output (3 detections per frame). The bar = the track with
# largest y-variance over the clip; track 0 is the bar in 96% of clips. 34% of
# OHP trajectories contain at least one NaN frame (likely occlusion at the
# press peak) — Phase 6 needs track-0 default + NaN interpolation.

# %%
import os, zipfile, json, random, collections
import numpy as np
import matplotlib.pyplot as plt
ROOT = '/content/drive/MyDrive/Fitness-AQA_dataset_release'

def load_traj_samples(zip_rel, n=200, seed=0):
    out = {}
    with zipfile.ZipFile(os.path.join(ROOT, zip_rel)) as zf:
        names = sorted(x for x in zf.namelist() if x.lower().endswith('.json'))
        random.Random(seed).shuffle(names)
        for nm in names[:n]:
            out[os.path.basename(nm)] = json.loads(zf.read(nm))
    return out

squat_tr = load_traj_samples('Squat/Unlabeled_Dataset/bar_trajectories_raw.zip', n=200)
ohp_tr   = load_traj_samples('OHP/Unlabeled_Dataset/bar_trajectories_raw.zip',   n=200)

# Squat: flat list of y-centers — direct.
squat_bar = {k: np.array([float(x) for x in v], dtype=float) for k, v in squat_tr.items()}

# OHP: per-frame list of K detections, each [[x1,y1,x2,y2,conf]].
def track_y_series(clip, k):
    s = []
    for fr in clip:
        if k < len(fr):
            bb = fr[k]
            inner = bb[0] if (isinstance(bb, list) and bb and isinstance(bb[0], list)) else bb
            if isinstance(inner, list) and len(inner) >= 4:
                s.append((float(inner[1]) + float(inner[3])) / 2.0); continue
        s.append(np.nan)
    return np.array(s, dtype=float)

def pick_bar_track(clip):
    K = len(clip[0]) if clip else 0
    series = [track_y_series(clip, k) for k in range(K)]
    variances = [np.nanvar(s) if np.isfinite(np.nanvar(s)) else -1.0 for s in series]
    best = int(np.argmax(variances)) if series else 0
    return series[best], best

ohp_bar, picks = {}, collections.Counter()
for k, clip in ohp_tr.items():
    s, idx = pick_bar_track(clip); ohp_bar[k] = s; picks[idx] += 1

for lbl, S in [('Squat', squat_bar), ('OHP', ohp_bar)]:
    lens = [len(s) for s in S.values()]
    nan_frac = np.mean([bool(np.isnan(s).any()) for s in S.values()])
    print(f'{lbl}: n={len(S)}  frames min {min(lens)} max {max(lens)} '
          f'mean {np.mean(lens):.1f} median {int(np.median(lens))} | '
          f'{nan_frac*100:.0f}% have any NaN')
print('OHP track-index picked as the bar:', dict(picks))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (lbl, S) in zip(axes, [('Squat', squat_bar), ('OHP', ohp_bar)]):
    for k in list(S)[:6]:
        s = S[k]
        rng = np.nanmax(s) - np.nanmin(s)
        if not np.isfinite(rng) or rng == 0: continue
        norm = (s - np.nanmin(s)) / rng
        ax.plot(np.linspace(0, 1, len(norm)), norm, alpha=0.7)
    ax.set_title(f'{lbl} — 6 barbell trajectories (height-normalized)')
    ax.set_xlabel('clip progress'); ax.set_ylabel('bar height (norm)'); ax.invert_yaxis()
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Cell 6 — sample-frame grids: Squat KIE (mid-rep) and Shallow

# %%
import os, zipfile, json, random, tempfile, cv2
import numpy as np
import matplotlib.pyplot as plt
ROOT = '/content/drive/MyDrive/Fitness-AQA_dataset_release'
def jload(*p): return json.load(open(os.path.join(ROOT, *p)))

kie     = jload('Squat/Labeled_Dataset/Labels/error_knees_inward.json')
shallow = jload('Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/labels_shallow_depth.json')

random.seed(1)
kie_pos = random.sample([k for k, v in kie.items() if v], 4)
kie_neg = random.sample([k for k, v in kie.items() if not v], 4)
sh_pos  = random.sample([k for k, v in shallow.items() if int(v) == 1], 4)
sh_neg  = random.sample([k for k, v in shallow.items() if int(v) == 0], 4)

def midframe(zip_rel, fname):
    with zipfile.ZipFile(os.path.join(ROOT, zip_rel)) as zf:
        data = zf.read('videos/' + fname)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tf:
        tf.write(data); p = tf.name
    cap = cv2.VideoCapture(p)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release(); os.unlink(p)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None

def load_img(zip_rel, fname):
    with zipfile.ZipFile(os.path.join(ROOT, zip_rel)) as zf:
        data = zf.read('crops_unaligned/' + fname + '.jpg')
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def grid(title, imgs, ids):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for i, (im, idd) in enumerate(zip(imgs, ids)):
        ax = axs[i // 2, i % 2]
        if im is not None: ax.imshow(im)
        ax.set_title(idd, fontsize=9); ax.axis('off')
    fig.suptitle(title, fontsize=12); plt.tight_layout(); plt.show()

ZSQ = 'Squat/Labeled_Dataset/videos.zip'
ZSH = 'Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/images.zip'
grid('Squat — KIE+ (knees inward labelled)  · mid-rep frame',
     [midframe(ZSQ, k + '.mp4') for k in kie_pos], kie_pos)
grid('Squat — KIE- (no KIE)                  · mid-rep frame',
     [midframe(ZSQ, k + '.mp4') for k in kie_neg], kie_neg)
grid('Squat — Shallow+ (depth error)         · image crop',
     [load_img(ZSH, k) for k in sh_pos], sh_pos)
grid('Squat — Shallow- (good depth)          · image crop',
     [load_img(ZSH, k) for k in sh_neg], sh_neg)

# %% [markdown]
# ## Cell 7 — KIE+ & KFE+ at error-interval midpoint, KFE- mid-rep

# %%
import os, zipfile, json, random, tempfile, cv2
import matplotlib.pyplot as plt
ROOT = '/content/drive/MyDrive/Fitness-AQA_dataset_release'
def jload(*p): return json.load(open(os.path.join(ROOT, *p)))

kie = jload('Squat/Labeled_Dataset/Labels/error_knees_inward.json')
kfe = jload('Squat/Labeled_Dataset/Labels/error_knees_forward.json')
random.seed(2)
kie_pos = random.sample([k for k, v in kie.items() if v], 4)
kfe_pos = random.sample([k for k, v in kfe.items() if v], 4)
kfe_neg = random.sample([k for k, v in kfe.items() if not v], 4)

ZSQ, FPS = 'Squat/Labeled_Dataset/videos.zip', 30.0

def grab(fname, label_dict=None, clip_id=None):
    """If label_dict & clip_id given -> frame at longest error interval midpoint, else clip mid-rep."""
    with zipfile.ZipFile(os.path.join(ROOT, ZSQ)) as zf:
        data = zf.read('videos/' + fname)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tf:
        tf.write(data); p = tf.name
    cap = cv2.VideoCapture(p)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if label_dict is not None:
        iv = max(label_dict[clip_id], key=lambda x: x[1] - x[0])
        idx = max(0, min(int(round(0.5 * (iv[0] + iv[1]) * FPS)), n - 1))
    else:
        idx = n // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release(); os.unlink(p)
    return (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None), idx

def grid(title, items):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for i, (im, cap_txt) in enumerate(items):
        ax = axs[i // 2, i % 2]
        if im is not None: ax.imshow(im)
        ax.set_title(cap_txt, fontsize=9); ax.axis('off')
    fig.suptitle(title, fontsize=12); plt.tight_layout(); plt.show()

items = []
for k in kie_pos:
    img, idx = grab(k + '.mp4', kie, k); items.append((img, f'{k}  (err-frame {idx})'))
grid('Squat — KIE+ at error-interval midpoint', items)

items = []
for k in kfe_pos:
    img, idx = grab(k + '.mp4', kfe, k); items.append((img, f'{k}  (err-frame {idx})'))
grid('Squat — KFE+ at error-interval midpoint', items)

items = []
for k in kfe_neg:
    img, idx = grab(k + '.mp4'); items.append((img, f'{k}  (mid-rep frame {idx})'))
grid('Squat — KFE- (no KFE) at mid-rep frame', items)
