import zipfile, os, shutil

repo    = r"C:\Users\tsh_x\Desktop\FitNova Application"
stage   = os.path.join(os.environ['TEMP'], 'fitnova_src_stage')
out_zip = os.path.join(repo, 'fitnova_src.zip')

if os.path.isdir(stage):
    shutil.rmtree(stage)
os.makedirs(stage)

items = [
    # Package init files
    'backend/__init__.py',
    'backend/services/__init__.py',
    'backend/training/__init__.py',
    # Training scripts
    'backend/training/train_form_model.py',
    'backend/training/train_ssl_pretrain.py',   # SSL pretraining (Cell 7)
    # Subpackages (all .py files, no __pycache__)
    'backend/training/models',
    'backend/training/preprocessing',
    'backend/training/evaluation',   # reality_check, offline_eval, visualize_pipeline
    # Services (form_analyzer imported by evaluation scripts)
    'backend/services/form_analyzer.py',
    'backend/services/form_session.py',
    # No model weights in src zip — Colab trains fresh;
    # SSL encoder weights are generated in Cell 7 on Colab itself.
]

for item in items:
    src = os.path.join(repo, item.replace('/', os.sep))
    dst = os.path.join(stage, item.replace('/', os.sep))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    elif os.path.isfile(src):
        shutil.copy2(src, dst)
    print('  +', item)

# Write zip with forward-slash arc names
with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(stage):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for fname in files:
            if fname.endswith('.pyc'):
                continue
            abs_path = os.path.join(root, fname)
            arc_name = os.path.relpath(abs_path, stage).replace(os.sep, '/')
            zf.write(abs_path, arc_name)

# Verify
with zipfile.ZipFile(out_zip) as zf:
    names = zf.namelist()
    sep = '\\'
    ok = all(sep not in n for n in names)
    assert 'backend/training/train_form_model.py' in names
    assert 'backend/training/train_ssl_pretrain.py' in names
    assert 'backend/services/form_analyzer.py' in names
    assert 'backend/services/form_session.py' in names
    assert 'backend/training/evaluation/reality_check.py' in names
    assert ok, "backslashes still present"
    print("All forward slashes: OK")
    for n in sorted(names):
        print(' ', n)

shutil.rmtree(stage)
print("\nfitnova_src.zip: %.1f MB" % (os.path.getsize(out_zip) / 1e6))
print("Done. Re-upload fitnova_src.zip to Colab.")
