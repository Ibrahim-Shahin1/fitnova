import pickle, time
from backend.services.content_filter import ContentBasedFilter

cf = ContentBasedFilter()
print(f'Loaded {cf.n_programs} programs')

# Test: Beginner + Strength + 1hr sessions + 4 days/week
t = time.time()
results = cf.get_top_n({'experience_level': 1, 'workout_type': 'Strength', 'session_duration_hours': 1.0, 'workout_frequency': 4})
elapsed = (time.time()-t)*1000
print(f'Scored in {elapsed:.0f}ms')

catalog = pickle.load(open('backend/data/program_catalog.pkl', 'rb'))
print('\nTop 10 for Beginner/Strength:')
for pid in results[:10]:
    p = catalog[pid]
    print(f'  {pid}: {p["title"][:50]} | level={p["level_encoded"]} str={p["has_strength"]} tpw={p["time_per_workout_minutes"]}min')

# Test 2: Expert + Yoga
results2 = cf.get_top_n({'experience_level': 3, 'workout_type': 'Yoga', 'session_duration_hours': 1.5, 'workout_frequency': 5})
print('\nTop 10 for Expert/Yoga:')
for pid in results2[:10]:
    p = catalog[pid]
    print(f'  {pid}: {p["title"][:50]} | level={p["level_encoded"]} yoga={p["has_yoga"]} tpw={p["time_per_workout_minutes"]}min')
