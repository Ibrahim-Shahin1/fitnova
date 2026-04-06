import pickle
import time

from backend.services.content_filter import ContentBasedFilter

cf = ContentBasedFilter()
print(f'Loaded {cf.n_programs} programs')
catalog = pickle.load(open('backend/data/program_catalog.pkl', 'rb'))

# --- Test 1: Beginner + Strength + 1hr + 4 days/week ---
t = time.time()
results = cf.get_top_n({
    'experience_level': 1,
    'workout_type': 'Strength',
    'session_duration_hours': 1.0,
    'workout_frequency': 4,
})
elapsed = (time.time() - t) * 1000
print(f'\nTest 1 — Beginner/Strength (scored in {elapsed:.0f}ms)')

print('Top 10:')
for pid in results[:10]:
    p = catalog[pid]
    print(f'  {pid}: {p["title"][:50]} | level={p["level_encoded"]} str={p["has_strength"]}')

# Assertions
assert elapsed < 100, f"Scoring too slow: {elapsed:.0f}ms (expected <100ms)"
top5_levels = [catalog[pid]['level_encoded'] for pid in results[:5]]
assert all(l == 0 for l in top5_levels), f"Top-5 should all be Beginner (level=0), got: {top5_levels}"
top5_strength = [catalog[pid]['has_strength'] for pid in results[:5]]
assert all(s == 1 for s in top5_strength), f"Top-5 should all have has_strength=1, got: {top5_strength}"
print('  [PASS] All top-5 are Beginner Strength programs')
print(f'  [PASS] Latency {elapsed:.0f}ms < 100ms')

# --- Test 2: Expert + Yoga + 1.5hr + 5 days/week ---
results2 = cf.get_top_n({
    'experience_level': 3,
    'workout_type': 'Yoga',
    'session_duration_hours': 1.5,
    'workout_frequency': 5,
})
print('\nTest 2 — Expert/Yoga')
print('Top 10:')
for pid in results2[:10]:
    p = catalog[pid]
    print(f'  {pid}: {p["title"][:50]} | level={p["level_encoded"]} yoga={p["has_yoga"]} str={p["has_strength"]}')

top5_yoga = [catalog[pid]['has_yoga'] for pid in results2[:5]]
assert any(y == 1 for y in top5_yoga), f"At least some top-5 should have has_yoga=1, got: {top5_yoga}"
print(f'  [PASS] Top-5 yoga flags: {top5_yoga}')

print('\n[ALL TESTS PASSED]')
