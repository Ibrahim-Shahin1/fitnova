import pickle, time

print("Loading recommender (this loads TensorFlow + NeuMF model)...")
t0 = time.time()
from backend.services.recommender import Recommender
rec = Recommender()
print(f"Loaded in {time.time()-t0:.1f}s\n")

catalog = pickle.load(open('backend/data/program_catalog.pkl', 'rb'))

# Test 1: Beginner Strength user
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
print(f"=== Beginner/Strength (scored in {elapsed:.0f}ms) ===")
import pandas as pd
uf = pd.read_csv('backend/data/user_features.csv')
uid1 = result['similar_user_id']
mapped1 = uf[uf['user_id'] == uid1].iloc[0]
print(f"Mapped to trained user_id: {uid1} (exp={int(mapped1['experience_level'])}, wt={int(mapped1['workout_type_encoded'])})")
print(f"Top program: {result['program_id']}")

p = catalog[result['program_id']]
print(f"  Title: {p['title']}")
print(f"  Level: {p['level_encoded']} | Strength: {p['has_strength']}")

print(f"\nContent filter top-5: {result['content_candidates'][:5]}")
print(f"NeuMF re-ranked top-5:")
for pid, score in result['neumf_ranked'][:5]:
    pp = catalog[pid]
    print(f"  {pid}: {pp['title'][:45]} | score={score:.4f}")

# Test 2: Expert Yoga user
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
print(f"=== Expert/Yoga (scored in {elapsed2:.0f}ms) ===")
uid2 = result2['similar_user_id']
mapped2 = uf[uf['user_id'] == uid2].iloc[0]
print(f"Mapped to trained user_id: {uid2} (exp={int(mapped2['experience_level'])}, wt={int(mapped2['workout_type_encoded'])})")
print(f"Top program: {result2['program_id']}")

p2 = catalog[result2['program_id']]
print(f"  Title: {p2['title']}")
print(f"  Level: {p2['level_encoded']} | Yoga: {p2['has_yoga']}")

print(f"\nNeuMF re-ranked top-5:")
for pid, score in result2['neumf_ranked'][:5]:
    pp = catalog[pid]
    print(f"  {pid}: {pp['title'][:45]} | score={score:.4f}")

# Check that NeuMF changes the ordering vs content filter
cf_order = result['content_candidates'][:10]
neumf_order = [pid for pid, _ in result['neumf_ranked'][:10]]
print(f"\n{'='*50}")
print(f"Content filter top-10: {cf_order}")
print(f"NeuMF re-ranked top-10: {neumf_order}")
print(f"Order changed: {cf_order != neumf_order}")
