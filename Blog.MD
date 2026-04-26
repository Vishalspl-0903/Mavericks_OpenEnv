

**MedFlow: Training AI to Triage Emergencies Like Clinical Experts**

Every second in an emergency department matters. Clinicians must assign triage urgency levels (ESI 1-5) under pressure and with incomplete information. MedFlow turns that problem into a training loop for a small language model, so it can learn when to classify and when to ask for more detail.

**A Small Model, a Real Triage Problem**

The notebook loads `unsloth/Qwen2.5-1.5B-Instruct`, adds LoRA adapters, and trains with GRPO. It starts with a quick health check, then builds a client around the OpenEnv triage server. The setup is intentionally lean: one model, one environment, and one safety goal.

**Three Cases, One Lesson**

The training set revolves around three cases: classic MI, suspected meningitis, and masked sepsis. They look similar on the surface, but the correct response changes fast when the vitals, history, or hidden details shift. That is what makes the notebook interesting: the model has to reason, not just label.

![MedFlow triage system overview](System.png)

This visual shows MedFlow handling emergency patients with real-time triage decisions under uncertainty.
It highlights the core loop: assign urgency, request clarifying details, and prioritize high-risk cases safely.


**How the Reward Works**

The reward function is built around safety. Exact ESI matches are best, near misses still count a little, and undertriage gets hit hard. The notebook also rewards useful reasoning, penalizes broken formatting, and gives extra credit when the model asks for clarification in the right cases. In short: safety first, confidence second.

**Training in Stages**

The notebook trains in three prompt stages. Stage 0 is the baseline, Stage 1 adds structured reasoning steps, and Stage 2 adds few-shot examples. That progression matters: the model improves from rough triage to more careful, expert-like decisions.

**What Improved Most**

Across the notebook, the trained model does better on all three tasks, with the biggest jump on masked sepsis. That is the most useful signal in the whole project: the harder the case, the more the staged prompts help.

**The Hard Test**

The notebook does not stop at clean examples. It also stress-tests silent MI, afebrile meningitis, and masked sepsis with near-normal vitals. That adversarial check is the right move for triage work, because it exposes where a model can still undertriage.

**Why It Stands Out**

The notebook keeps the goal simple: train a model to make safer emergency decisions under uncertainty. It is not trying to sound smart. It is trying to avoid dangerous mistakes, ask better questions, and escalate the right patients faster.

---