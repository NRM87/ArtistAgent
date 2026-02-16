# TODO / Roadmap

This file tracks planned work that should preserve the core intent of the project:
an evolving artist agent with strong self-reflection and autonomous creative behavior.

## User-Requested Next Work

- [ ] Refine the self-reflection loop weighting across run stages so the artist feels more actively evolving.
  - [ ] Revisit weighting per stage:
    - vision generation
    - iteration prompt refinement
    - critique/judgment
    - soul revision
  - [ ] Make weighting behavior explicit, inspectable, and easy to tune per profile/artist.

- [ ] Add cross-artist gallery reviews.
  - [ ] Let artists view works from other artists' galleries.
  - [ ] Let artists write structured reviews on other artists' works.
  - [ ] Persist review artifacts with author/target/timestamp and link to reviewed artwork.

- [ ] Add a review-ingestion run mode.
  - [ ] New run type where an artist reads received reviews.
  - [ ] Artist evaluates review merit (accept/reject/partial).
  - [ ] Artist updates soul/memories when it decides reviews are valuable.

## Suggested Enhancements (Optional)

- [ ] Add a lightweight "artist season" timeline (phases/eras) derived from memory history and review signals.
- [ ] Add periodic "challenge prompts" (self-imposed constraints) generated from soul drift to force stylistic exploration.
- [ ] Add simple visual stats per artist (score trend, novelty trend, tier distribution) to make evolution observable.
- [ ] Add collaboration mode:
  - [ ] one artist creates
  - [ ] one critiques
  - [ ] creator decides whether to incorporate critique and logs why

