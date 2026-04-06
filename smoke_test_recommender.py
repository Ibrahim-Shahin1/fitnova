import sys
import pickle
import time

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("Loading recommender (loads TensorFlow + NeuMF model)...")
t0 = time.time()
from backend.services.recommender import Recommender
rec = Recommender()
print(f"Loaded in {time.time()-t0:.1f}s\n")

with open('backend/data/program_catalog.pkl', 'rb') as handle:
    catalog = pickle.load(handle)
uf = pd.read_csv('backend/data/user_features.csv')

# --- Test 1: Beginner/Strength ---
profile1 = {
    'experience_level': 1,
    'workout_type': 'Strength',
    'session_duration_hours': 1.0,
    'workout_frequency': 4,
    'age': 22,
    'gender': 'Male',
    'bmi': 24.0,
}

t = time.time()
result = rec.recommend(profile1)
elapsed = (time.time() - t) * 1000

uid1 = result['similar_user_id']
mapped1 = uf[uf['user_id'] == uid1].iloc[0]
print(f"=== Beginner/Strength (scored in {elapsed:.0f}ms) ===")
print(f"Mapped to user_id: {uid1} (exp={int(mapped1['experience_level'])}, wt={int(mapped1['workout_type_encoded'])})")
p = catalog[result['program_id']]
print(f"Top program: {result['program_id']} — {p['title']}")
print(f"  Level={p['level_encoded']} | Strength={p['has_strength']}")
print(f"NeuMF top-5:")
for pid, score in result['neumf_ranked'][:5]:
    pp = catalog[pid]
    print(f"  {pid}: {pp['title'][:45]} | score={score:.4f}")

# Assertions
assert elapsed < 2000, f"Pipeline too slow: {elapsed:.0f}ms (expected <2000ms)"
assert int(mapped1['experience_level']) == 1, \
    f"Cold-start mapped Beginner to exp={int(mapped1['experience_level'])} (expected 1)"
assert catalog[result['program_id']]['level_encoded'] == 0, \
    f"Top program should be Beginner (level=0)"
assert catalog[result['program_id']]['has_strength'] == 1, \
    f"Top program should have has_strength=1"
print("  [PASS] Cold-start mapped to Beginner user")
print("  [PASS] Top program is Beginner/Strength")
print(f"  [PASS] Latency {elapsed:.0f}ms < 2000ms")

# --- Test 2: Expert/Yoga ---
print(f"\n{'='*50}")
profile2 = {
    'experience_level': 3,
    'workout_type': 'Yoga',
    'session_duration_hours': 1.5,
    'workout_frequency': 5,
    'age': 35,
    'gender': 'Female',
    'bmi': 22.0,
}

t = time.time()
result2 = rec.recommend(profile2)
elapsed2 = (time.time() - t) * 1000

uid2 = result2['similar_user_id']
mapped2 = uf[uf['user_id'] == uid2].iloc[0]
print(f"=== Expert/Yoga (scored in {elapsed2:.0f}ms) ===")
print(f"Mapped to user_id: {uid2} (exp={int(mapped2['experience_level'])}, wt={int(mapped2['workout_type_encoded'])})")
p2 = catalog[result2['program_id']]
print(f"Top program: {result2['program_id']} — {p2['title']}")
print(f"  Level={p2['level_encoded']} | Yoga={p2['has_yoga']} | Primary={p2.get('primary_type')}")
print(f"NeuMF top-5:")
for pid, score in result2['neumf_ranked'][:5]:
    pp = catalog[pid]
    print(f"  {pid}: {pp['title'][:45]} | primary={pp.get('primary_type')} | score={score:.4f}")

assert int(mapped2['experience_level']) == 3, \
    f"Cold-start mapped Expert to exp={int(mapped2['experience_level'])} (expected 3)"
assert p2['has_yoga'] == 1, "Top Yoga recommendation should have has_yoga=1"
assert p2.get('primary_type') == 'Yoga', \
    f"Top Yoga recommendation should be Yoga-primary, got: {p2.get('primary_type')}"
top5_yoga_primary = sum(catalog[pid].get('primary_type') == 'Yoga' for pid, _ in result2['neumf_ranked'][:5])
assert top5_yoga_primary >= 3, \
    f"Expected at least 3 Yoga-primary programs in top-5, got: {top5_yoga_primary}"
print("  [PASS] Cold-start mapped to Expert user")
print("  [PASS] Top Yoga recommendation preserves Yoga intent")

# --- NeuMF reordering check ---
cf_order = result['content_candidates'][:10]
neumf_order = [pid for pid, _ in result['neumf_ranked'][:10]]
print(f"\n{'='*50}")
print(f"Content filter top-10: {cf_order}")
print(f"NeuMF re-ranked top-10: {neumf_order}")
assert cf_order != neumf_order, "NeuMF should change the ordering vs content filter"
print("  [PASS] NeuMF changes the ordering")

print('\n[ALL TESTS PASSED]')
