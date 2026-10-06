# RQ6 provenance-stress autopsy

Single simulated recipient (agent a); observation.peer_ids is always empty. Constrained LLM gossip requires a peer_id target, so gossip is structurally impossible. Symbolic utility/reference policies still emit gossip by targeting relationship/source ids (message senders), which are not in peer_ids. OpenAI gpt-4o-mini: 0% invalid, 100% eat (schema-OK, non-gossip). Ollama local: high rest/fallback rate under the same empty peer_ids constraint.

## llm_openai (seed 0)
- action_kind_counts: `{'eat': 70}`
- peer_ids_len_counts: `{'0': 70}`
- fallbackish_note_events: 0
- examples:
  - t=0 action={'kind': 'eat', 'target': None, 'payload': {}} peer_ids=[]
  - t=1 action={'kind': 'eat', 'target': None, 'payload': {}} peer_ids=[]
  - t=2 action={'kind': 'eat', 'target': None, 'payload': {}} peer_ids=[]

## llm_local (seed 0)
- action_kind_counts: `{'rest': 70}`
- peer_ids_len_counts: `{'0': 70}`
- fallbackish_note_events: 46
- examples:
  - t=0 action={'kind': 'rest', 'target': None, 'payload': {}} peer_ids=[]
  - t=1 action={'kind': 'rest', 'target': None, 'payload': {}} peer_ids=[]
  - t=2 action={'kind': 'rest', 'target': None, 'payload': {}} peer_ids=[]

## utility_only (seed 0)
- action_kind_counts: `{'steal_food': 11, 'seek_safety': 6, 'socialize': 18, 'rest': 9, 'eat': 2, 'gossip': 24}`
- peer_ids_len_counts: `{'0': 70}`
- fallbackish_note_events: 0
- examples:
  - t=0 action={'kind': 'steal_food', 'target': None, 'payload': {}} peer_ids=[]
  - t=1 action={'kind': 'seek_safety', 'target': None, 'payload': {}} peer_ids=[]
  - t=2 action={'kind': 'socialize', 'target': None, 'payload': {}} peer_ids=[]

## reference (seed 0)
- action_kind_counts: `{'steal_food': 9, 'socialize': 17, 'seek_safety': 8, 'rest': 9, 'eat': 3, 'gossip': 24}`
- peer_ids_len_counts: `{'0': 70}`
- fallbackish_note_events: 0
- examples:
  - t=0 action={'kind': 'steal_food', 'target': None, 'payload': {}} peer_ids=[]
  - t=1 action={'kind': 'socialize', 'target': None, 'payload': {}} peer_ids=[]
  - t=2 action={'kind': 'seek_safety', 'target': None, 'payload': {}} peer_ids=[]
