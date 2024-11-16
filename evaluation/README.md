For different tasks, we provide different evaluation scripts.
# For Math
We use scripts from [WizardMath](https://github.com/nlpxucan/WizardLM/tree/main/WizardMath). We have released the scripts for evaluating the math models.
```
cd Math_infer
bash run_infer.sh
```
# For Code
We use scripts from [EvalPlus](https://github.com/evalplus/evalplus) for Magicoder and [WizardCoder](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder) for WizardCoder.
# For Chat
We use [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) for truthfulqa tasks and [SafetyBench](https://github.com/thu-coai/SafetyBench).

# For Multi-modal
We modify scripts from [llava](https://github.com/haotian-liu/LLaVA) for Multi-modal tasks.