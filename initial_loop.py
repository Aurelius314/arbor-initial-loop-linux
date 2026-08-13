'''
end-to-end solution for COP

initial simpleevol-style agent loop:

problem + compressed history -> LLM solution -> evaluator -> feedback
'''

import json
import logging
from pathlib import Path
import time
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class SimpleEvol:
    def __init__(
        self,
        cfg,
        root_dir,
        client,
        eval_dataset=None,
        eval_tool=None,
        problem_name="tsp",
        obj_type="min",
        exp_records_dir=None,
    ):
        self.cfg = cfg
        self.client = client

        # ========= Generic config =========
        self.problem_name = problem_name
        if obj_type not in {"min", "max"}:
            raise ValueError("obj_type must be 'min' or 'max'")
        self.obj_type = obj_type
        self.problem_warn = "Visit every city exactly once. Do not repeat any city, except that the starting city may appear once again as the final city to close the tour."
        self.max_experiments = cfg.max_experiments
        self.compress_every = cfg.compress_every
        self.max_generation_attempts = getattr(
            cfg, "max_generation_attempts", self.max_experiments * 3
        )
        if self.max_experiments < 1:
            raise ValueError("max_experiments must be >= 1")
        if self.max_generation_attempts < self.max_experiments:
            raise ValueError("max_generation_attempts must be >= max_experiments")

        if eval_dataset is None:
            raise ValueError("eval_dataset must be injected by the protected evaluator")
        self.eval_dataset = eval_dataset
        if not isinstance(self.eval_dataset, list) or not self.eval_dataset:
            raise ValueError("eval_dataset must contain a non-empty list")

        self.exp_records_dir = Path(exp_records_dir) if exp_records_dir else None
        if self.exp_records_dir is not None:
            self.exp_records_dir.mkdir(parents=True, exist_ok=True)

        # ========= Prompts =========
        self.problem_tai_prompts = self._load_batch_prompts(
            self.eval_dataset, range(len(self.eval_dataset))
        )

        # ========= Runtime states =========
        self.current_instance_index = 0
        self.current_instance = self.eval_dataset[0]
        self.problem_size = int(self.current_instance["num_nodes"])
        self.problem_tai_prompt = self.problem_tai_prompts[0]
        if eval_tool is None:
            raise ValueError("eval_tool must be injected by the protected evaluator")
        self.eval_tool = eval_tool
        self.messages = self._build_initial_messages()
        self.experiment_results = []

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # -------------------------------------------------------------------------
    # Prompt / message construction
    # -------------------------------------------------------------------------

    def _load_batch_prompts(self, eval_dataset, batch_indices) -> list[str]:
        """
        Prepare batch prompts for the model.

        Args:
            eval_dataset: The evaluation dataset
            batch_indices (list): List of indices for the current batch

        Returns:
            list: Batch prompts
        """
        alpaca_prompt = """Below is an instruction describing a combinatorial optimization problem. It is paired with an input that provides the data of the instance.
        Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective.

        ### Instruction:{}

        ### Input:{}

        ### Response:"""

        batch_prompts = []
        for idx in batch_indices:
            instruction = eval_dataset[idx]['instruction']
            user_input = eval_dataset[idx]['input']
            prompt = alpaca_prompt.format(instruction, user_input)
            batch_prompts.append(prompt)

        return batch_prompts

    def _build_initial_messages(self) -> str:

        tai_prompt = self.problem_tai_prompt.strip()

        system_generator_prompt = """
            You are an optimization expert specializing in end-to-end solution construction for combinatorial optimization problems.
            Your task is to independently construct and evaluate complete solutions over at most {max_experiments} experiments.
            Through these experiments, discover effective construction principles and task-specific knowledge.

            ## Experiment Requirements

            After each experiment:
                **Record**: the construction idea and evaluation result.
                **Reflect**: analyze why the strategy worked or failed.
                **Compare**: compare its performance with previous experiments.
                **Plan**: think about what direction to try next

            Experiments should test meaningful hypotheses.
            If a strategy repeatedly fails, explore a substantially different construction principle.

            1. When you are doing summarization:
            All summaries are concise working notes for future exploration (previous context will be cleared after summarization to save tokens).
            Therefore, avoid unnecessary wording and focus only on conclusions, insights, and hypotheses that influence future exploration.
            Do not include any previous feasible solution or implementation details in summaries.

            2. When you are generating the solution:
            Output exactly one complete candidate solution, and a concise description of the construction strategy.
            """.format(max_experiments=self.max_experiments).strip()

        format_suffix = """

        Important output rule:
        Your response must contain exactly two parts in the following order:
        1. A single candidate solution formatted as above requirement.

        2. After generating the solution, write a concise description after the solution (you should wrap it using <description> ... </description>).
        """.strip()

        system_prompt = f"{system_generator_prompt}\n\n{format_suffix}"

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{tai_prompt}\n\nImportant feasibility warning:\n{self.problem_warn}"
                ),
            },
        ]

    # 响应中没有 Route: [...]
    def _build_retry_message_for_bad_format(self):
        return {
            "role": "user",
            "content": (
                f"Invalid solution format. Output the components required in ### Input: "
                f", followed by `<description>...</description>`.\n\n"
                f"Important feasibility warning:\n{self.problem_warn}"
            ),
        }

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------
    def get_best_result(self,experiment_results):
        valid_results = [r for r in experiment_results if r.get("obj") is not None]
        if not valid_results:
            return None
        if self.obj_type == "min":
            return min(valid_results, key=lambda x: x["obj"])
        else:
            return max(valid_results, key=lambda x: x["obj"])


    def _call_llm_once(self, messages):
        """
        ReEvo-style normal chat completion, no tool calling.
        Assumes client.chat_completion(...) returns a list-like choices object.
        """
        responses = self.client.chat_completion(
            n=1,
            messages=messages,
        )
        return responses[0].message

    # -------------------------------------------------------------------------
    # Context compression
    # -------------------------------------------------------------------------
    def compress_context(self):

        logger.info("#" * 70)
        logger.info(f"[Context Compression] {self.experiment_count} experiments completed, generating intermediate summary...")
        logger.info("#" * 70)

        SUMMARY_USER_PROMPT = """
        Please provide a concise **intermediate summary** of the {experiment_count} end-to-end solution attempts conducted so far, including:

        1. **Attempt history**: Briefly summarize the construction strategy and evaluation result (objective value) of each attempt. Similar attempts with similar results may be grouped.
        2. **Key findings**: Explain how different construction principles performed and the likely reasons.
        3. **Mistakes to avoid**: Record ineffective reasoning patterns, invalid-output patterns, and recurring feasibility errors.
        4. **Task knowledge**: Summarize reusable insights learned about the problem structure and effective solution construction.

        Important constraints:
        - Do NOT include complete or partial tours, node sequences, solution fragments, or any previously generated feasible solution.
        - The summary must only preserve abstract strategies, evaluation results, mistakes, and task-level insights.
        - Do not suggest a specific next solution or next-step modification.

        Keep the summary within 300 words.
        """.strip()

        summary_request = {
            "role": "user",
            "content": SUMMARY_USER_PROMPT.format(experiment_count=self.experiment_count),
        }

        summary_messages = list(self.messages) + [summary_request]
        summary_message = self._call_llm_once(summary_messages)
        summary_content = summary_message.content or ""

        logger.info("[Intermediate Summary]")
        logger.info("-" * 40)
        logger.info(summary_content)

        initial_messages = self.messages[:2]
        remaining_experiments = self.max_experiments - self.experiment_count

        # best_code_section = ""
        compressed_messages = initial_messages + [
            {
                "role": "assistant",
                "content": (
                    f"[Intermediate Summary - {self.experiment_count}/{self.max_experiments} "
                    f"experiments completed]\n\n{summary_content}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Based on your intermediate summary, please continue exploring.\n\n"
                    f"Experiment progress: {self.experiment_count}/{self.max_experiments} completed, "
                    f"{remaining_experiments} remaining.\n\n"
                    f"Important feasibility warning:\n{self.problem_warn}\n\n"
                    "Output the next candidate solution using exactly the required format."
                ),
            },
        ]

        logger.info(f"[Context compressed] messages: Length {len(self.messages)} -> length {len(compressed_messages)}")
        logger.info("#" * 70)

        self.messages = compressed_messages

    def _extract_description_from_response(self, response_text: str) -> str:
        if not response_text:
            return ""
        match = re.search(
            r"<description>(.*?)</description>",
            response_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    def _extract_solution_and_description(self, text: str):
        """
        Given a full decoded text which contains:
        ### Instruction (Problem): ...
        ### Input (Instance Data): ...
        ### Response (Proposed Solution): ...
        Return only the substring after '### Response (Proposed Solution):'
        so that we can parse route, objective, etc., from that substring.
        """

        match = re.search(r"Route\s*:\s*\[([^\]]*)\]", text, flags=re.IGNORECASE)
        if not match:
            return None, ""
        try:
            solution_text = match.group(1).strip()
            solution = (
                [int(x.strip()) for x in solution_text.split(",")]
                if solution_text
                else []
            )
        except ValueError:
            return None, ""

        description = self._extract_description_from_response(text)
        description = self._normalize_description(description)

        return solution, description

    def _evaluate_solution(self, solution):
        return self.eval_tool.evaluate(solution, self.current_instance)

    def _load_optimal_objective(self) -> float:
        return float(self.eval_tool.reference_objective(self.current_instance_index))

    def _calculate_optimality_gap(self, candidate_obj, optimal_obj):
        if candidate_obj is None or optimal_obj == 0:
            return None
        if self.obj_type == "min":
            return (candidate_obj - optimal_obj) / abs(optimal_obj) * 100.0
        return (optimal_obj - candidate_obj) / abs(optimal_obj) * 100.0

    def _normalize_description(self, text: str, max_chars: int = 2000) -> str:
        if not text:
            return ""
        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n...[truncated]"
        return text

    def _save_experiment_record(self, record: dict) -> Path:
        """Persist one direct-solution experiment, following utils.py's format."""
        if self.exp_records_dir is None:
            return None
        instance_dir = (
            self.exp_records_dir
            / self.timestamp
            / f"instance_{self.current_instance_index:03d}"
        )
        instance_dir.mkdir(parents=True, exist_ok=True)
        record_path = instance_dir / f"exp_{record['experiment']:03d}.txt"

        content = [
            f"Experiment: {record['experiment']}",
            f"Instance: {self.current_instance_index}",
            f"Problem: {self.problem_name}",
            f"Problem size: {self.problem_size}",
            "",
            "=== Evaluation Result ===",
            f"feasible: {record['feasible']}",
            f"candidate_obj: {record['obj']}",
            f"optimal_obj: {record['optimal_obj']}",
            f"optimality_gap: {record['optimality_gap']}",
            f"exec_time: {record['time']}",
            f"error: {record['error']}",
            "",
            "=== Description ===",
            record["description"],
            "",
            "=== Solution ===",
            f"Solution: {record['solution']}",
        ]
        record_path.write_text("\n".join(content), encoding="utf-8")
        return record_path
    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    def _request_final_summary(self):
        final_user_msg = {
            "role": "user",
            "content": (
                f"You have completed all {self.max_experiments} experiments. "
                "Please now provide your final summary:\n\n"
                "1. What strategies did you try?\n"
                "2. How did each strategy perform?\n"
                "3. What patterns or insights did you discover?\n"
                "4. What types of construction designs are more effective?\n"
                "5. Which abstract construction strategy performed best?"
            ),
        }
        self.messages.append(final_user_msg)

        final_message = self._call_llm_once(self.messages)
        self.messages.append(
            {
                "role": "assistant",
                "content": final_message.content or "",
            }
        )

        logger.info("=" * 70)
        logger.info("[LLM Final Summary]")
        logger.info("=" * 70)
        logger.info(final_message.content if final_message.content else "")

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    def evolve(self):
        """Run ``max_experiments`` independently for every dev instance."""
        best_solutions = []
        best_objectives = []

        for instance_index, instance in enumerate(self.eval_dataset):
            self.current_instance_index = instance_index
            self.current_instance = instance
            self.problem_size = int(instance["num_nodes"])
            self.problem_tai_prompt = self.problem_tai_prompts[instance_index]

            coords = instance.get("instance")
            if not isinstance(coords, list) or len(coords) != self.problem_size:
                coordinate_count = len(coords) if isinstance(coords, list) else 0
                raise ValueError(
                    f"dev instance {instance_index} declares {self.problem_size} nodes "
                    f"but contains {coordinate_count} coordinates"
                )

            # Every instance is an independent optimization run.
            self.messages = self._build_initial_messages()
            self.experiment_results = []
            best_solution, best_objective = self._evolve_current_instance()
            best_solutions.append(best_solution)
            best_objectives.append(best_objective)

        # Preserve the historical API for the common one-instance dev file.
        if len(self.eval_dataset) == 1:
            return best_solutions[0], best_objectives[0]
        return best_solutions, best_objectives

    def _evolve_current_instance(self):
        logger.info("=" * 70)
        logger.info("SimpleEvol started.")
        logger.info(f"Problem: {self.problem_name}")
        logger.info(
            f"Dev instance: {self.current_instance_index + 1}/{len(self.eval_dataset)}"
        )
        logger.info(f"Problem size: {self.problem_size}")
        logger.info(f"Max experiments: {self.max_experiments}")
        logger.info(f"Obj Type: {self.obj_type}")
        logger.info(f"Context compression: every {self.compress_every} evals" if self.compress_every > 0
                    else "Context compression: disabled")
        logger.info("=" * 70)

        self.experiment_count = 0
        generation_attempts = 0
        last_compress_at = 0
        optimal_obj = self._load_optimal_objective()
        while (
            self.experiment_count < self.max_experiments
            and generation_attempts < self.max_generation_attempts
        ):
            generation_attempts += 1

            logger.info("=" * 70)
            logger.info(f"[Iteration {self.experiment_count}] Completed {self.experiment_count}/{self.max_experiments} evaluations")
            logger.info("=" * 70)

            assistant_message = self._call_llm_once(self.messages)
            assistant_content = assistant_message.content or ""

            solution, description = self._extract_solution_and_description(assistant_content)
            if solution is None:
                logger.warning("No valid solution extracted from model response.")
                self.messages.append(self._build_retry_message_for_bad_format())
                continue

            if not description:
                description = "Construction description unavailable."

            self.messages.append(
                {
                    "role": "assistant",
                    "content": f"[Candidate Description]\n{description}",
                }
            )

            self.experiment_count += 1

            logger.info("-" * 40)
            logger.info(
                f"[Experiment {self.experiment_count}/{self.max_experiments}] Running evaluation..."
            )

            start_time = time.time()
            evaluation = self._evaluate_solution(solution)
            exec_time = time.time() - start_time
            feasible = evaluation["feasible"]
            candidate_obj = evaluation["obj"]
            error_msg = evaluation["error"]
            optimality_gap = self._calculate_optimality_gap(candidate_obj, optimal_obj)

            logger.info("Evaluation result:")
            logger.info(f"  feasible: {feasible}")
            logger.info(f"  candidate obj: {candidate_obj}")
            logger.info(f"  optimal obj: {optimal_obj}")
            logger.info(f"  optimality gap: {optimality_gap}")
            logger.info(f"  time: {exec_time}")
            logger.info(f"  error: {error_msg}")

            record = {
                "experiment": self.experiment_count,
                "solution": solution,
                "description": description,
                "feasible": feasible,
                "obj": candidate_obj,
                "optimal_obj": optimal_obj,
                "optimality_gap": optimality_gap,
                "time": exec_time,
                "error": error_msg,
            }
            self.experiment_results.append(record)
            record_path = self._save_experiment_record(record)
            logger.info(f"Experiment record saved: {record_path}")

            # 加入evaluate结果
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your last candidate solution has been evaluated.\n\n"
                        f"Evaluation result:\n"
                        f"Experiment {self.experiment_count} feasibility: {feasible}; "
                        f"candidate objective: "
                        f"{f'{candidate_obj:.6f}' if candidate_obj is not None else 'N/A'}; "
                        f"evaluation error: {error_msg if error_msg is not None else 'None'}. "
                        f"Please analyze the result internally and output your next candidate solution.\n\n"
                        f"Important feasibility warning:\n{self.problem_warn}"
                    ),
                }
            )

            # compress context if needed
            if (
                self.compress_every > 0
                and self.experiment_count < self.max_experiments
                and self.experiment_count >= last_compress_at + self.compress_every
            ):
                self.compress_context()
                last_compress_at = self.experiment_count

        if self.experiment_count < self.max_experiments:
            logger.warning(
                "Stopped after %s generation attempts with %s/%s valid experiments.",
                generation_attempts,
                self.experiment_count,
                self.max_experiments,
            )

        self._request_final_summary()

        # ------------------------------------------------------------------
        # Post-run logging
        # ------------------------------------------------------------------
        logger.info("=" * 70)
        logger.info("Experimental Statistics")
        logger.info("=" * 70)

        best_result = self.get_best_result(self.experiment_results)
        feasible_count = sum(r["feasible"] for r in self.experiment_results)
        feasibility_rate = feasible_count / self.max_experiments
        best_objective = best_result["obj"] if best_result is not None else None
        best_optimality_gap = self._calculate_optimality_gap(best_objective, optimal_obj)

        logger.info(
            f"Feasibility Rate: {feasibility_rate:.2%} "
            f"({feasible_count}/{self.max_experiments})"
        )
        logger.info(f"Best obj: {best_objective}")
        logger.info(f"Optimality Gap: {best_optimality_gap}")

        if self.experiment_results:
            logger.info(f"Completed {len(self.experiment_results)} experiments.")
            for r in self.experiment_results:
                status = "✓" if r["obj"] is not None else "✗"
                obj_str = f"{r['obj']:.6f}" if r["obj"] is not None else "N/A"
                logger.info(f"[{status}] Experiment {r['experiment']}: obj={obj_str}")

            valid_results = [r for r in self.experiment_results if r["obj"] is not None]
            if valid_results:
                if self.obj_type == "min":
                    worst = max(valid_results, key=lambda x: x["obj"])
                else:
                    worst = min(valid_results, key=lambda x: x["obj"])
                logger.info(f"Best result: Exp. {best_result['experiment']}, obj={best_result['obj']:.6f}")
                logger.info(f"Worst result: Exp. {worst['experiment']}, obj={worst['obj']:.6f}")
                if worst["obj"] != 0:
                    if self.obj_type == "min":
                        improvement = ((worst["obj"] - best_result["obj"]) / worst["obj"]) * 100
                    else:
                        improvement = -((worst["obj"] - best_result["obj"]) / worst["obj"]) * 100
                    logger.info(f"Improvement span: {improvement:.2f}%")
        else:
            logger.info("No experiments were completed.")

        # Optional client-side token summary
        if hasattr(self.client, "get_usage_summary"):
            logger.info("=" * 70)
            logger.info("Client Usage Summary")
            logger.info("=" * 70)
            try:
                logger.info(json.dumps(self.client.get_usage_summary(), indent=2, ensure_ascii=False))
            except Exception:
                logger.info(str(self.client.get_usage_summary()))


        best_solution = best_result["solution"] if best_result is not None else []
        best_objective = best_result["obj"] if best_result is not None else None

        return best_solution, best_objective
