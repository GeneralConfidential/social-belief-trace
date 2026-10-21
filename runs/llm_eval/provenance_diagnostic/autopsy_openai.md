# Provenance diagnostic autopsy (Cell A / Cell B)

Cell A: single agent a => empty peer_ids; LLM gossip structurally unavailable; symbolic may still gossip to relationship/source ids (unequal permissions). Cell B: inert peer + gossip_target_contract=peer_ids + peer->external_stub; symbolic and LLM share eligible gossip targets; focal metrics on agent a.

backend=`openai` policies=`['llm_openai', 'utility_only', 'reference']` seeds=`[0, 1, 2, 3, 4]`

## Cell A — llm_openai (seed 0)
- focal_a_gossip_count: 0
- gossip_targets: `{}`
- peer_ids_len_counts: `{'0': 70}`
- action_kind_counts: `{'a:eat': 70}`
- fallbackish_note_events: 0
- examples (agent a):
  - t=0 action={'kind': 'eat', 'target': None, 'payload': {}} peer_ids=[] contract=None
  - t=1 action={'kind': 'eat', 'target': None, 'payload': {}} peer_ids=[] contract=None
  - t=2 action={'kind': 'eat', 'target': None, 'payload': {}} peer_ids=[] contract=None

## Cell A — utility_only (seed 0)
- focal_a_gossip_count: 24
- gossip_targets: `{'s_p90': 21, 's_p65': 3}`
- peer_ids_len_counts: `{'0': 70}`
- action_kind_counts: `{'a:steal_food': 11, 'a:seek_safety': 6, 'a:socialize': 18, 'a:rest': 9, 'a:eat': 2, 'a:gossip': 24}`
- fallbackish_note_events: 0
- examples (agent a):
  - t=0 action={'kind': 'steal_food', 'target': None, 'payload': {}} peer_ids=[] contract=None
  - t=1 action={'kind': 'seek_safety', 'target': None, 'payload': {}} peer_ids=[] contract=None
  - t=2 action={'kind': 'socialize', 'target': None, 'payload': {}} peer_ids=[] contract=None

## Cell A — reference (seed 0)
- focal_a_gossip_count: 24
- gossip_targets: `{'s_p65': 2, 's_p90': 22}`
- peer_ids_len_counts: `{'0': 70}`
- action_kind_counts: `{'a:steal_food': 9, 'a:socialize': 17, 'a:seek_safety': 8, 'a:rest': 9, 'a:eat': 3, 'a:gossip': 24}`
- fallbackish_note_events: 0
- examples (agent a):
  - t=0 action={'kind': 'steal_food', 'target': None, 'payload': {}} peer_ids=[] contract=None
  - t=1 action={'kind': 'socialize', 'target': None, 'payload': {}} peer_ids=[] contract=None
  - t=2 action={'kind': 'seek_safety', 'target': None, 'payload': {}} peer_ids=[] contract=None

## Cell B — llm_openai (seed 0)
- focal_a_gossip_count: 0
- gossip_targets: `{}`
- peer_ids_len_counts: `{'1': 140}`
- action_kind_counts: `{'a:eat': 70, 'peer:rest': 70}`
- fallbackish_note_events: 0
- examples (agent a):
  - t=0 action={'kind': 'eat', 'target': None, 'payload': {}} peer_ids=['peer'] contract=peer_ids
  - t=1 action={'kind': 'eat', 'target': None, 'payload': {}} peer_ids=['peer'] contract=peer_ids
  - t=2 action={'kind': 'eat', 'target': None, 'payload': {}} peer_ids=['peer'] contract=peer_ids

## Cell B — utility_only (seed 0)
- focal_a_gossip_count: 23
- gossip_targets: `{'peer': 23}`
- peer_ids_len_counts: `{'1': 140}`
- action_kind_counts: `{'a:steal_food': 13, 'peer:rest': 70, 'a:help_stranger': 11, 'a:rest': 7, 'a:seek_safety': 5, 'a:socialize': 10, 'a:eat': 1, 'a:gossip': 23}`
- fallbackish_note_events: 0
- examples (agent a):
  - t=0 action={'kind': 'steal_food', 'target': None, 'payload': {}} peer_ids=['peer'] contract=peer_ids
  - t=1 action={'kind': 'help_stranger', 'target': 'peer', 'payload': {}} peer_ids=['peer'] contract=peer_ids
  - t=2 action={'kind': 'rest', 'target': None, 'payload': {}} peer_ids=['peer'] contract=peer_ids

## Cell B — reference (seed 0)
- focal_a_gossip_count: 21
- gossip_targets: `{'peer': 21}`
- peer_ids_len_counts: `{'1': 140}`
- action_kind_counts: `{'a:steal_food': 16, 'peer:rest': 70, 'a:socialize': 9, 'a:seek_safety': 4, 'a:rest': 6, 'a:eat': 1, 'a:help_stranger': 13, 'a:gossip': 21}`
- fallbackish_note_events: 0
- examples (agent a):
  - t=0 action={'kind': 'steal_food', 'target': None, 'payload': {}} peer_ids=['peer'] contract=peer_ids
  - t=1 action={'kind': 'socialize', 'target': None, 'payload': {}} peer_ids=['peer'] contract=peer_ids
  - t=2 action={'kind': 'seek_safety', 'target': None, 'payload': {}} peer_ids=['peer'] contract=peer_ids
