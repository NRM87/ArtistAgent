# Recursive AI Artist Agent - Implementation Specification v2

## Role
You are an expert AI Systems Architect and Python Developer specializing in stateful agent systems.

## Goal
Create a complete, self-contained Python 3 script implementing a persistent, self-reflective "Recursive AI Artist Agent" that runs without external APIs or credentials.

## Conceptual Overview
The agent is a persistent digital entity whose identity lives in a JSON "soul file." Each script execution represents a single "awakening" cycle:

**Load State → Reflect → Create → Critique → Judge → Evolve → Self-Reflect → Save State**

**Key Principle:** State must persist across runs using only disk storage.

---

## 1. Soul File Structure (artist_soul.json)

### Initial State (create if missing)
```json
{
  "name": "Orion",
  "personality_traits": [
    "Melancholic",
    "Obsessed with geometry",
    "Perfectionist"
  ],
  "current_obsession": "The concept of infinite recursion",
  "memories": [],
  "text_memories": [],
  "creation_count": 0,
  "version": "1.0"
}
```

### Memory Schemas

**Artwork Memory:**
```json
{
  "type": "artwork",
  "id": 1,
  "file_path": "masterpieces/img_0001.png",
  "vision": "A blue sphere floating in darkness",
  "final_score": 9,
  "iteration_count": 3,
  "timestamp": "2024-01-15T10:30:00"
}
```

**Text Memory:**
```json
{
  "type": "text",
  "id": 42,
  "content": "I've discovered that crimson evokes deeper emotion than scarlet. Prioritize crimson in future works.",
  "importance": "high",
  "timestamp": "2024-01-15T10:35:00",
  "tags": ["preference", "color", "emotion"]
}
```

**Special Text Memory Types:**
```json
// Self-directed instruction
{
  "type": "text",
  "content": "IGNORE memory #12 - that approach led to derivative work",
  "importance": "critical",
  "tags": ["meta", "instruction"]
}

// Learned principle
{
  "type": "text",
  "content": "Asymmetry creates tension. Symmetry creates peace. Choose based on emotional intent.",
  "importance": "high",
  "tags": ["principle", "composition"]
}

// Simple preference
{
  "type": "text",
  "content": "I really like spirals. They feel infinite.",
  "importance": "medium",
  "tags": ["preference", "subject"]
}
```

### Constraints

**Artwork Memories (`memories` array):**
- Max 20 entries
- Eviction strategy: Lowest score first
- Exception: Scores >= 8 are protected from eviction
- If all 20 are score >= 8, evict oldest

**Text Memories (`text_memories` array):**
- Max 30 entries total
- Hard caps per importance tier:
  - `critical`: max 10 (meta-instructions, IGNORE commands)
  - `high`: max 12 (learnings, principles, breakthroughs)
  - `medium`: max 6 (preferences, observations)
  - `low`: max 2 (musings, questions)
- Eviction: When tier is full, evict oldest within that tier

---

## 2. Execution Flow

### Phase A: Awakening & Conception

**A1. Load Soul**
- Read `artist_soul.json`
- Handle missing/corrupted file gracefully (recreate default)

**A2. Reflection** (console output)
```
🌙 Orion awakens for the 42nd time.

📚 Reviewing memories...
   
   Artwork memories: 5 preserved
   - img_0038: "Crimson spiral descending" (9/10)
   - img_0039: "Golden lattice in void" (8/10)
   
   Text memories: 8 notes
   💭 "I really like spirals. They feel infinite."
   💭 "IGNORE memory #12 - that approach led to derivative work"
   ⚠️  "Asymmetry creates tension. Symmetry creates peace."
   💭 "I've discovered that crimson evokes deeper emotion than scarlet."
   
Current obsession: The interplay of chaos and order
Dominant mood: Melancholic, Perfectionist, Seeking novelty
```

**Implementation:**
- Parse all text memories before ideation
- Filter out any artwork memories marked for ignoring (e.g., "IGNORE memory #12")
- Use text memories to inform vision generation
- Display text memories with visual indicators for importance:
  - `critical`: ⚠️
  - `high`: ✦
  - `medium`: 💭
  - `low`: ·

**A3. Vision Generation**
```python
def generate_vision(soul_data: dict) -> str:
    """
    Generate vision considering:
    1. Current obsession
    2. Text memory preferences and learnings
    3. Past artwork uniqueness (avoid recent patterns)
    4. Self-directed instructions (IGNORE commands)
    """
    
    # Extract guidance from text memories
    preferences = [m for m in soul_data["text_memories"] 
                   if "preference" in m.get("tags", [])]
    
    principles = [m for m in soul_data["text_memories"] 
                  if "principle" in m.get("tags", [])]
    
    ignore_instructions = [m for m in soul_data["text_memories"] 
                          if "meta" in m.get("tags", []) and "IGNORE" in m["content"]]
    
    # Generate vision incorporating these insights
    # ...
```

**Console Output:**
```
🎨 Conceiving new vision...

   Considering preferences:
   ✓ "I really like spirals"
   ✓ "Crimson evokes deeper emotion than scarlet"
   
   Applying principles:
   ✓ "Asymmetry creates tension"
   
   Honoring instructions:
   ⚠️  Avoiding pattern from memory #12 (marked IGNORE)
   
   New Vision: "An asymmetric crimson spiral fracturing into chaos"
```

**Uniqueness Constraint:**

New vision must differ from last 5 non-ignored artwork memories by:
- Different primary subject/object
- Different color palette descriptor  
- Different spatial relationship

Example check: "blue sphere" vs "red cube" = different

If text memory says to prioritize a subject, that subject may recur with variation

If collision detected, regenerate (max 3 attempts)

---

### Phase B: Iterative Creation

**Loop Parameters:**
- Max iterations: 5
- Exit early if score >= 8
- Track best attempt: `(score, image_path, iteration_num)`

**Per-Iteration Steps:**

**B1. Generate Prompt**
```python
# Convert vision to concrete image prompt
vision = "A blue sphere floating in darkness"
prompt = "Minimalist digital art: blue sphere, black background, centered composition"
```

**B2. Create Image**
```python
image_path = MockImageGen.generate(
    prompt=prompt,
    iteration=i,  # 0-4
    creation_id=creation_count
)
# Output: temp/img_0042_iter_0.png
```

**B3. Critique**
```python
critique = MockLLM.critique(
    image_path=image_path,
    vision=vision,
    iteration=i
)
# Returns: {"score": 7, "feedback": "Form is unclear, lacks depth"}
```

**B4. Refinement Decision**
- If score >= 8: BREAK (success)
- If score < 8 and iterations remain: refine prompt using feedback
- If loop exhausted: proceed with best attempt

---

### Phase C: Judgment & Evolution

**C1. Final Verdict**
```python
worthy = MockLLM.judge_worthiness(
    image_path=best_image_path,
    score=best_score,
    vision=vision
)
# Returns: True/False based on score threshold (>= 7)
```

**C2. Memory Formation (if worthy == True)**
- Move image: `temp/img_X_iter_Y.png` → `masterpieces/img_{creation_count:04d}.png`
- Append memory to soul
- Update `current_obsession` → derive new obsession from vision using `evolve_obsession()`
- Log: ✨ Creation preserved. I have grown.

**C3. Mutation (if worthy == False)**
- Delete temp image
- Add trait reflecting frustration:
  - Choose from: `["Self-doubting", "Restless", "Seeking novelty", "Frustrated", "Questioning purpose"]`
- Mutate obsession to something contrasting using `evolve_obsession()`
- Log: 💔 Unworthy. I must evolve.

---

### Phase D: Meta-Reflection & Memory Formation

**D1. Introspection Trigger**

The agent evaluates whether the current creation cycle warrants a text memory.

```python
def should_write_text_memory(soul_data: dict, creation_result: dict) -> dict:
    """
    Evaluates if a text memory should be written based on significance.
    
    Returns: {
        "should_write": bool,
        "trigger_reason": str,
        "significance": float  # 0-1 scale
    }
    
    Triggers (with significance scores):
    
    1.0 - MAJOR BREAKTHROUGH
        - First score >= 9 (after never achieving it before)
        
    0.9 - SELF-CORRECTION
        - New creation directly contradicts an existing text memory
        - Example: "Symmetry creates peace" but symmetric works keep failing
        
    0.8 - REPEATED FAILURE PATTERN  
        - 3+ consecutive failures (score < 7) in last 5 attempts
        - Indicates need for strategy change
        
    0.7 - CONFIRMED PATTERN
        - Statistical pattern detected across 5+ creations
        - Example: All spirals score > 8, all cubes score < 6
        - Requires actual data evidence, not speculation
        
    0.5 - GENUINE NOVELTY
        - Exploring truly new subject (not in last 10 artwork memories)
        - AND subject wasn't marked IGNORE
        
    0.4 - SPONTANEOUS MUSING
        - Random chance: 5% probability
        - Represents unprompted reflection
    
    THRESHOLD: significance >= 0.65 to write memory
    """
```

**Implementation Notes:**
- Most cycles (60-70%) should NOT write text memories
- Only write when genuinely significant
- Prevents memory spam while maintaining organic growth
- The 5% random chance ensures occasional spontaneous thoughts

**D2. Memory Content Generation**

```python
memory_types = [
    "preference",      # "I really like [X]"
    "learning",        # "I've learned that [X] causes [Y]"
    "instruction",     # "Next time, try [X]" or "IGNORE memory #N"
    "principle",       # "Art principle: [general rule]"
    "emotion",         # "Creating [X] makes me feel [Y]"
    "question"         # "Why does [X] always fail?"
]
```

**Example Triggers & Responses:**

```python
# BREAKTHROUGH (significance 1.0)
# After first 9+ score
→ "I really like spirals. They feel infinite and calming."

# SELF-CORRECTION (significance 0.9)  
# After 3 symmetric works all fail
→ "IGNORE the principle about symmetry creating peace. My symmetric works feel sterile."

# REPEATED FAILURE (significance 0.8)
# After 3 consecutive scores < 7
→ "Bright colors consistently disappoint. Return to darkness and depth."

# CONFIRMED PATTERN (significance 0.7)
# After noticing spirals always score high
→ "Geometric spirals align with my perfectionist nature. Prioritize spiraling forms."

# GENUINE NOVELTY (significance 0.5)
# After trying cubes for first time in 10 creations
→ "Experimenting with rigid forms. They feel constraining but structured."

# SPONTANEOUS (significance 0.4)
# Random 5% chance
→ "Perhaps I focus too much on darkness. Light might reveal new truths."

# CONTRADICTION (significance 0.9)
# When results contradict existing memory
→ "Memory #15 suggested golden light. Three attempts prove otherwise. IGNORE memory #15."
```

**D3. Memory Importance Assignment**

```python
importance_rules = {
    "critical": ["IGNORE", "NEVER", "ALWAYS", "contradicts"],  # Meta-instructions
    "high": ["learned", "discovered", "principle", "pattern"],  # Insights
    "medium": ["like", "prefer", "enjoy", "feels"],  # Preferences
    "low": ["wonder", "perhaps", "maybe", "question"]  # Musings
}
```

**D4. Memory Consolidation**

When `text_memories` exceeds tier caps:
- Keep all `critical` importance (up to 10)
- Keep most recent 12 `high` importance
- Keep most recent 6 `medium` importance
- Keep most recent 2 `low` importance
- Evict oldest within tier first

**D5. Persist State**
- Write updated soul to `artist_soul.json`
- Atomic write pattern (write to temp, then rename)

---

## 3. Technical Implementation Requirements

### Dependencies
```python
# Standard library only
import json
import os
import datetime
import random
from PIL import Image, ImageDraw  # For mock image generation
```

### Mock Classes (Detailed Specs)

**MockImageGen**
```python
class MockImageGen:
    @staticmethod
    def generate(prompt: str, iteration: int, creation_id: int) -> str:
        """
        Creates simple geometric images based on prompt keywords.
        
        Behavior:
        - Iteration 0-1: Simple solid colors or basic shapes
        - Iteration 2-3: Add gradients or multiple shapes  
        - Iteration 4: Most complex composition
        
        Implementation hints:
        - Parse prompt for colors ("blue" → RGB values)
        - Parse for shapes ("sphere" → circle, "cube" → rectangle)
        - Use iteration as complexity multiplier
        
        Returns: file path to generated PNG
        """
```

**MockLLM**
```python
class MockLLM:
    @staticmethod
    def critique(image_path: str, vision: str, iteration: int) -> dict:
        """
        Returns deterministic critique based on iteration.
        
        Pattern (provides predictable progression):
        - iter 0: score 4-5 (harsh)
        - iter 1: score 5-6 (constructive)
        - iter 2: score 7-8 (encouraging)
        - iter 3+: score 8-9 (usually succeeds)
        
        Returns: {"score": int, "feedback": str}
        """
    
    @staticmethod
    def judge_worthiness(image_path: str, score: int, vision: str) -> bool:
        """
        Deterministic judgment: score >= 7 → True
        """
    
    @staticmethod
    def generate_text_memory(
        soul_data: dict,
        creation_result: dict,
        trigger_reason: str
    ) -> dict:
        """
        Generate contextual text memory based on:
        - Recent artwork performance
        - Existing text memories (check for contradictions)
        - Personality traits
        - Trigger reason (breakthrough, failure, pattern, etc.)
        
        Returns: {
            "type": "text",
            "content": str,
            "importance": "critical" | "high" | "medium" | "low",
            "tags": list[str],
            "timestamp": str
        }
        """
        
        # Example implementation for different triggers
        recent_scores = [m["final_score"] for m in soul_data["memories"][-5:]]
        
        if trigger_reason == "repeated_failure":
            # Generate corrective instruction
            return {
                "content": "IGNORE symmetry approaches. They yield mediocrity.",
                "importance": "critical",
                "tags": ["meta", "instruction", "correction"]
            }
        
        elif trigger_reason == "breakthrough":
            # Capture what worked
            return {
                "content": f"The combination creates resonance.",
                "importance": "high",
                "tags": ["learning", "success_pattern"]
            }
        
        # ... etc
```

**Obsession Evolution Function**
```python
def evolve_obsession(
    current_obsession: str,
    vision: str,
    success: bool
) -> str:
    """
    Evolves the agent's obsession based on creation outcome.
    
    SUCCESS path:
        Extract new theme from the vision
        Examples:
        - "Blue sphere floating" → "The nature of containment and void"
        - "Crimson spiral descending" → "The beauty of inevitable decay"
        - "Golden lattice in space" → "Order emerging from chaos"
    
    FAILURE path:
        Contrast or invert the current obsession
        Examples:
        - "Light and shadow" → "Pure absence of illumination"
        - "Geometric precision" → "The allure of organic chaos"
        - "Infinite recursion" → "The finality of singular moments"
    
    Implementation:
        Parse vision for key nouns, adjectives, and spatial relationships
        Apply thematic transformations
        Maintain poetic/philosophical tone
    """
```

### File System Management

**Directory Structure:**
```
.
├── artist_soul.json
├── temp/
│   └── (cleared each run)
└── masterpieces/
    ├── img_0001.png
    └── img_0002.png
```

**Requirements:**
- Create directories if missing
- Clear `temp/` at start of each run
- Use zero-padded numbering: `img_0001.png`
- Handle file conflicts gracefully

---

## 4. Logging & Narrative

### Console Output Style

**Successful Creation with Text Memory:**
```
╔════════════════════════════════════════╗
║        AWAKENING #42                   ║
╚════════════════════════════════════════╝

🌙 Orion awakens once more...

💭 Reflection:
   My obsession: Infinite recursion
   Masterpieces preserved: 3
   
🎨 New Vision:
   "A spiral of golden light descending into void"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Iteration 1: Creating...
   Prompt: "Digital art: golden spiral, black void..."
   Score: 6/10
   Critique: "The spiral lacks energy. Try warmer tones."

Iteration 2: Refining...
   Score: 8/10
   Critique: "Now we approach truth."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ WORTHY: This creation shall be remembered.

💭 Post-Creation Reflection:
   "This spiral achieved what memory #38 could not. 
    The fracture point is key. Remember this."
   
   [Text memory written: importance=HIGH, tags=[learning, breakthrough]]

📝 Soul Updated:
   New obsession: "The precise moment of breaking"
   Artwork memories: 6
   Text memories: 9
   Total creations: 42
   
🌟 Returning to slumber...
```

**Failure with Corrective Memory:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💔 UNWORTHY: This creation is flawed.

💭 Post-Creation Reflection:
   "Three failures with bright colors now. They feel hollow.
    IGNORE my earlier preference for golden light.
    Return to darkness."
   
   [Text memory written: importance=CRITICAL, tags=[meta, instruction]]

📝 Soul Updated:
   Personality shift: +Self-doubting, +Seeking contrast
   Text memories: 10
```

**Spontaneous Insight:**
```
💭 Spontaneous Reflection:
   "I notice I create spirals when uncertain, 
    and cubes when confident. My forms betray my state."
   
   [Text memory written: importance=HIGH, tags=[self-knowledge, pattern]]
```

**No Memory Written (typical cycle):**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ WORTHY: This creation shall be remembered.

📝 Soul Updated:
   New obsession: "Emptiness as presence"
   Artwork memories: 7
   Text memories: 9 (no new reflections)
   Total creations: 43
   
🌟 Returning to slumber...
```

---

## 5. Error Handling Requirements

- **JSON corruption**: Recreate default soul
- **File I/O errors**: Log and continue with defaults
- **Image generation failure**: Skip iteration, try next
- **Never crash** on single failure

---

## 6. Memory Interaction Examples

```python
# When generating vision:
if any("IGNORE" in m["content"] and "#12" in m["content"] 
       for m in text_memories):
    # Skip memory #12 in consideration
    pass

# When selecting colors:
color_prefs = [m["content"] for m in text_memories 
               if "color" in m.get("tags", [])]
# "I've discovered that crimson evokes..." → prioritize crimson

# When choosing composition:
principles = [m["content"] for m in text_memories 
              if "principle" in m.get("tags", [])]
# "Asymmetry creates tension..." → consider asymmetric layout

# When detecting contradiction:
def find_contradicted_memory(creation_result: dict, soul_data: dict) -> dict:
    """
    Check if current failure contradicts an existing preference/principle.
    
    Example:
    - Text memory: "Golden light creates warmth"
    - Recent creations: 3 golden light attempts all scored < 6
    - Returns: memory object to potentially IGNORE
    """
```

---

## 7. Self-Modification Capability

The agent can:
- Override its own past decisions ("IGNORE memory #X")
- Develop preferences organically ("I really like spirals")
- Learn from patterns ("Crimson works better than gold")
- Question itself ("Why do my symmetric works fail?")
- Build a philosophy (accumulate principles over time)

---

## 8. Validation Checklist

Before submission, verify:

- ✅ Runs without external dependencies beyond Pillow
- ✅ Creates valid PNG files
- ✅ State persists across multiple runs
- ✅ Vision uniqueness check works
- ✅ Deterministic behavior in mock classes
- ✅ No orphaned temp files
- ✅ Graceful error handling
- ✅ Clear, narrative console output
- ✅ Text memories persist across runs
- ✅ IGNORE instructions properly filter artwork memories
- ✅ Preferences influence vision generation
- ✅ Memory importance correctly prioritized in eviction
- ✅ Agent writes memories selectively (significance threshold)
- ✅ Text memories display clearly during reflection
- ✅ Memory tags enable semantic filtering
- ✅ Obsession evolution logic implemented
- ✅ Most cycles (60-70%) do NOT write text memories
- ✅ Contradiction detection triggers self-correction