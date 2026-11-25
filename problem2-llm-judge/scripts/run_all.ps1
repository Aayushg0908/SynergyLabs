# Reproduce every Problem 2 result. Run from problem2-llm-judge/:  .\scripts\run_all.ps1
# Calls already in logs/calls.jsonl are served from the replay cache (no API cost); set
# $env:LLM_JUDGE_REPLAY = "only" to rebuild all reports offline from the committed log.
$ErrorActionPreference = "Stop"
$judge = ".\.venv\Scripts\judge.exe"

& $judge run --variant mitigated --tag suite_mitigated                    # suite in -> report out
& $judge generate --config configs\gen_prompt_v1.yaml                    # A/B outputs
& $judge generate --config configs\gen_prompt_v2.yaml
& $judge generate --config configs\gen_model_small.yaml
& $judge generate --config configs\gen_qwen_v2.yaml                      # for the self-preference test
& $judge compare --tag ab                                                # prompt v1 vs v2
& $judge compare --a results\ab\outputs_prompt_v2.yaml --b results\ab\outputs_allam_v2.yaml `
    --name-a gpt-oss-20b --name-b allam-2-7b --tag ab_model              # model A vs model B
& $judge bias                                                            # naive vs mitigated, all biases
& $judge validate                                                        # gold agreement, test-retest, cross-family
