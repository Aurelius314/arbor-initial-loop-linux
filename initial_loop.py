import json
import logging
from pathlib import Path
import time
import re

logger = logging.getLogger(__name__)


class SimpleEvol:
    PROBLEM_SPECS = {
        "tsp": {
            "label": "Route",
            "schema": "a JSON list containing every 0-indexed node exactly once; an optional repeated start node may close the tour",
            "warning": "Visit every node exactly once and form a complete tour.",
        },
        "cvrp": {
            "label": "Routes",
            "schema": "a JSON list of routes; every route starts and ends at depot 0, and every customer appears exactly once overall",
            "warning": "Cover every customer exactly once and respect vehicle capacity on every route.",
        },
        "op": {
            "label": "Route",
            "schema": "a JSON list of distinct 0-indexed nodes beginning at the configured start node",
            "warning": "Do not repeat nodes and keep the route within the maximum travel length.",
        },
        "mis": {
            "label": "Set",
            "schema": "a JSON list of distinct 0-indexed vertices in the independent set",
            "warning": "No selected pair may share an edge.",
        },
        "mvc": {
            "label": "Set",
            "schema": "a JSON list of distinct 0-indexed vertices in the vertex cover",
            "warning": "Every graph edge must have at least one selected endpoint.",
        },
        "pfsp": {
            "label": "Order",
            "schema": "a JSON list that is a permutation of all 1-indexed jobs",
            "warning": "Include every 1-indexed job exactly once.",
        },
        "jssp": {
            "label": "Schedule",
            "schema": "a JSON list with one row per machine; each row is a permutation of all 0-indexed jobs",
            "warning": "Give one complete job permutation per machine; the combined orders must respect job precedence without cycles.",
        },
    }

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
        record_instance_index=None,
    ):
        self.cfg = cfg
        self.client = client
        # The protected evaluator may attach a callable after construction.
        # Trace failures must never affect solving or scoring.
        self.trace_sink = None

        # ========= Generic config =========
        self.problem_name = problem_name
        if obj_type not in {"min", "max"}:
            raise ValueError("obj_type must be 'min' or 'max'")
        self.obj_type = obj_type
        if self.problem_name not in self.PROBLEM_SPECS:
            raise ValueError(f"unsupported COP problem: {self.problem_name}")
        self.problem_spec = self.PROBLEM_SPECS[self.problem_name]
        self.problem_warn = self.problem_spec["warning"]
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
        self.record_instance_index = record_instance_index
        if self.exp_records_dir is not None:
            self.exp_records_dir.mkdir(parents=True, exist_ok=True)

        # ========= Prompts =========
        self.problem_tai_prompts = self._load_batch_prompts(
            self.eval_dataset, range(len(self.eval_dataset))
        )

        # ========= Runtime states =========
        self.current_instance_index = 0
        self.current_instance = self.eval_dataset[0]
        self.problem_size = self._problem_size(self.current_instance)
        self.problem_tai_prompt = self.problem_tai_prompts[0]
        if eval_tool is None:
            raise ValueError("eval_tool must be injected by the protected evaluator")
        self.eval_tool = eval_tool
        self.messages = self._build_initial_messages()
        self.experiment_results = []

    def _emit_trace(self, event: str, **payload) -> None:
        sink = getattr(self, "trace_sink", None)
        if not callable(sink):
            return
        try:
            sink(event, payload)
        except Exception as exc:
            logger.warning("Trajectory trace emission failed: %s", exc)

    def _record_invalid_response(self, response_text: str, attempt: int) -> None:
        """Persist an unparsable model response for offline debugging only."""
        if self.exp_records_dir is None:
            return
        try:
            invalid_dir = self.exp_records_dir / "invalid_response"
            invalid_dir.mkdir(parents=True, exist_ok=True)
            record_path = invalid_dir / (
                f"attempt_{attempt:03d}_{time.time_ns()}.txt"
            )
            record_path.write_text(response_text, encoding="utf-8")
            logger.info("Invalid model response saved: %s", record_path)
        except Exception as exc:
            # Diagnostics must never change solving or scoring behavior.
            logger.warning("Failed to save invalid model response: %s", exc)

    # -------------------------------------------------------------------------
    # Prompt / message construction
    # -------------------------------------------------------------------------

    def _problem_size(self, instance) -> int:
        size_key = "n" if self.problem_name in {"pfsp", "jssp"} else "num_nodes"
        if size_key not in instance:
            raise ValueError(f"{self.problem_name} instance is missing {size_key}")
        return int(instance[size_key])

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

    def _build_initial_messages(self) -> list[dict[str, str]]:

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
            Never copy a previous candidate or expose any complete or partial candidate in the description or summaries.
            """.format(max_experiments=self.max_experiments).strip()

        format_suffix = """

        Important output rule:
        Return exactly one JSON object and no Markdown or surrounding text:
        {{"solution": <problem-specific JSON>, "description": "concise construction strategy"}}

        `solution` must have this problem-specific meaning:
        {solution_schema}

        The description must NOT include any current or previous solution, solution fragment, node/job sequence, route, set, assignment, or schedule.
        Ignore any legacy response-format wording inside the instance instruction; the JSON object format above is mandatory.
        """.format(solution_schema=self.problem_spec["schema"]).strip()

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

    def _build_retry_message_for_bad_format(self, category="missing_solution"):
        return {
            "role": "user",
            "content": (
                f"Invalid solution format ({category}). Start a fresh candidate from "
                "scratch and return exactly one JSON object with no Markdown or "
                "surrounding text: "
                "{\"solution\": <problem-specific JSON>, "
                "\"description\": \"concise construction strategy\"}. "
                f"The solution must be {self.problem_spec['schema']}.\n\n"
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


    def _call_llm_once(self, messages, response_mode="text"):
        """
        ReEvo-style normal chat completion, no tool calling.
        Assumes client.chat_completion(...) returns a list-like choices object.
        """
        responses = self.client.chat_completion(
            n=1,
            messages=messages,
            response_mode=response_mode,
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
        - Do NOT include complete or partial solutions, node/job sequences, routes, sets, assignments, schedules, or any previously generated feasible solution.
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

        self._emit_trace(
            "context_summary",
            experiment=self.experiment_count,
            summary=summary_content,
            messages_before=len(self.messages),
        )

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
        if not text:
            return None, "", "empty_response"

        stripped = text.strip()
        description = ""
        solution = None
        saw_json_container = False

        def unpack(value):
            if isinstance(value, dict) and "solution" in value:
                return value.get("solution"), value.get("description", "")
            if isinstance(value, list):
                return value, ""
            return None, ""

        # Preferred envelope: one complete JSON object. A complete bare list is
        # also accepted for backward compatibility and then strictly validated
        # by the protected problem evaluator.
        try:
            value = json.loads(stripped)
            saw_json_container = True
            solution, description = unpack(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        tagged = re.search(
            r"<solution>\s*(.*?)\s*</solution>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if solution is None and tagged:
            saw_json_container = True
            try:
                solution = json.loads(tagged.group(1))
            except (json.JSONDecodeError, TypeError, ValueError):
                return None, "", "invalid_json"

        # Accept a single complete JSON value inside one fenced code block.
        if solution is None:
            fenced = re.fullmatch(
                r"\s*```(?:json)?\s*(.*?)\s*```\s*",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if fenced:
                saw_json_container = True
                try:
                    solution, description = unpack(json.loads(fenced.group(1)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    return None, "", "invalid_json"

        # Historical Route/Routes/Set/Order/Schedule response convention.
        if solution is None:
            legacy = re.search(
                rf"\b{re.escape(self.problem_spec['label'])}\s*:\s*",
                text,
                flags=re.IGNORECASE,
            )
            if legacy:
                saw_json_container = True
                try:
                    solution, _ = json.JSONDecoder().raw_decode(
                        text[legacy.end():].lstrip()
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    return None, "", "invalid_json"

        if not isinstance(solution, list):
            category = "invalid_json_shape" if saw_json_container else "missing_solution"
            return None, "", category

        if not description:
            description = self._extract_description_from_response(text)
        if not isinstance(description, str):
            description = ""
        description = self._normalize_description(description)

        return solution, description, "parsed"

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
        instance_dir = self.exp_records_dir
        instance_dir.mkdir(parents=True, exist_ok=True)
        record_path = instance_dir / f"exp_{record['experiment']:03d}.txt"

        content = [
            f"Experiment: {record['experiment']}",
            "Instance: "
            f"{self.current_instance_index if self.record_instance_index is None else self.record_instance_index}",
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
                "Provide qualitative strategy notes only. Do not state or infer "
                "experiment counts, feasibility rates, objectives, gaps, route counts, "
                "or any other numerical result; those statistics are generated "
                "deterministically from evaluator events.\n\n"
                "1. What strategies did you try?\n"
                "2. What qualitative patterns or insights did you discover?\n"
                "3. What construction designs appeared more or less effective?"
            ),
        }
        self.messages.append(final_user_msg)

        final_message = self._call_llm_once(self.messages)
        self._emit_trace(
            "strategy_notes",
            experiment=self.experiment_count,
            notes=final_message.content or "",
        )
        self.messages.append(
            {
                "role": "assistant",
                "content": final_message.content or "",
            }
        )

        logger.info("=" * 70)
        logger.info("[Unverified LLM Strategy Notes — qualitative only]")
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
            self.problem_size = self._problem_size(instance)
            self.problem_tai_prompt = self.problem_tai_prompts[instance_index]
            if "instance" not in instance:
                raise ValueError(f"{self.problem_name} instance {instance_index} lacks structured data")

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
        self._emit_trace(
            "trajectory_start",
            problem=self.problem_name,
            problem_size=self.problem_size,
            max_experiments=self.max_experiments,
            max_generation_attempts=self.max_generation_attempts,
            compress_every=self.compress_every,
            objective_type=self.obj_type,
        )
        logger.info("=" * 70)
        logger.info("SimpleEvol started.")
        logger.info(f"Problem: {self.problem_name}")
        if self.record_instance_index is None:
            logger.info(
                f"Dev instance: {self.current_instance_index + 1}/{len(self.eval_dataset)}"
            )
        else:
            logger.info(f"Dataset instance: {self.record_instance_index}")
        logger.info(f"Problem size: {self.problem_size}")
        logger.info(f"Max experiments: {self.max_experiments}")
        logger.info(f"Obj Type: {self.obj_type}")
        logger.info(f"Context compression: every {self.compress_every} evals" if self.compress_every > 0
                    else "Context compression: disabled")
        logger.info("=" * 70)

        self.experiment_count = 0
        generation_attempts = 0
        format_failures = 0
        consecutive_format_failures = 0
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

            assistant_message = self._call_llm_once(
                self.messages, response_mode="solution"
            )
            assistant_content = assistant_message.content or ""

            solution, description, parse_status = (
                self._extract_solution_and_description(assistant_content)
            )
            self._emit_trace(
                "generation_attempt",
                attempt=generation_attempts,
                experiment=self.experiment_count + 1 if solution is not None else None,
                parsed=solution is not None,
                parse_status=parse_status,
                description=description,
            )
            if solution is None:
                format_failures += 1
                consecutive_format_failures += 1
                logger.warning("No valid solution extracted from model response.")
                self._record_invalid_response(assistant_content, generation_attempts)
                self._emit_trace(
                    "invalid_format",
                    attempt=generation_attempts,
                    category=parse_status,
                    total=format_failures,
                    consecutive=consecutive_format_failures,
                )
                retry_message = self._build_retry_message_for_bad_format(parse_status)
                # Do not stack repeated correction messages. After two
                # consecutive failures, replace the prior correction with a
                # clean from-scratch request while preserving abstract history.
                if (
                    consecutive_format_failures >= 2
                    and self.messages
                    and self.messages[-1].get("role") == "user"
                    and self.messages[-1].get("content", "").startswith(
                        "Invalid solution format ("
                    )
                ):
                    self.messages[-1] = retry_message
                else:
                    self.messages.append(retry_message)
                continue

            consecutive_format_failures = 0
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
            self._emit_trace(
                "experiment_result",
                experiment=self.experiment_count,
                feasible=feasible,
                objective=candidate_obj,
                optimal_objective=optimal_obj,
                optimality_gap=optimality_gap,
                error=error_msg,
                description=description,
                execution_seconds=exec_time,
            )
            record_path = self._save_experiment_record(record)
            logger.info(f"Experiment record saved: {record_path}")

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
        logger.info("Verified Experimental Statistics (from structured evaluator events)")
        logger.info("=" * 70)

        best_result = self.get_best_result(self.experiment_results)
        feasible_count = sum(r["feasible"] for r in self.experiment_results)
        completed_count = len(self.experiment_results)
        feasibility_rate = feasible_count / completed_count if completed_count else 0.0
        best_objective = best_result["obj"] if best_result is not None else None
        best_optimality_gap = self._calculate_optimality_gap(best_objective, optimal_obj)

        logger.info(
            f"Feasibility Rate: {feasibility_rate:.2%} "
            f"({feasible_count}/{completed_count})"
        )
        logger.info(
            f"Completion Rate: {completed_count / self.max_experiments:.2%} "
            f"({completed_count}/{self.max_experiments})"
        )
        logger.info(f"Generation Attempts: {generation_attempts}")
        logger.info(f"Format Failures: {format_failures}")
        error_counts = {}
        for result in self.experiment_results:
            if result["feasible"]:
                continue
            error = str(result.get("error") or "unknown")
            error_counts[error] = error_counts.get(error, 0) + 1
        logger.info(f"Evaluation Errors: {json.dumps(error_counts, ensure_ascii=False)}")
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

        self._emit_trace(
            "trajectory_complete",
            generation_attempts=generation_attempts,
            experiments=self.experiment_count,
            feasible_experiments=feasible_count,
            best_experiment=best_result["experiment"] if best_result else None,
            best_objective=best_objective,
            best_optimality_gap=best_optimality_gap,
        )

        return best_solution, best_objective
