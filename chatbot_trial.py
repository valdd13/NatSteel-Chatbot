import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from openai import OpenAI

DEFAULT_OPENAI_KEY = (
    "sk-proj-bhcLSdjLUnlP-d3POL6r6X8Rjys5EEULMkCt-PFwVgHhmF27cxWLwxlVEbTDNFw_"
    "DpXnipBckXT3BlbkFJXvE-RJPH6KhxCtIxE7rH1wJ_D_YedoMFANa9aPk-p74SORUqLEliN"
    "GpHh6Td6LqXbW5KpBrS4A"
)


@dataclass
class HistoricalStats:
    load_groups: Dict[str, List[str]]
    combo_frequency: Counter
    pair_frequency: Counter
    item_attributes: Dict[str, Dict[str, Any]]  # ITEM_NO -> attributes dict


class SteelLoadingPlanner:
    """
    Learn historical steel loading patterns and generate new loading plans with OpenAI assistance.

    The planner uses the history stored in data.xlsx to derive frequent loading templates.
    When new steel items arrive, it can propose a baseline plan based on historical combinations
    and optionally refine it via an OpenAI model.
    """

    def __init__(
        self,
        data_path: str,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        top_k_combos: int = 10,
    ) -> None:
        self.data_path = data_path
        self.model = model
        self.top_k_combos = top_k_combos
        self.dataframe = self._load_data()
        self.history = self._build_history()
        self.client = self._init_openai_client(openai_api_key)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_excel(self.data_path)
        required_columns = {"ITEM_NO", "LOAD_NO"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"The dataset must contain the columns {required_columns}. Missing: {missing}"
            )

        df["ITEM_NO"] = df["ITEM_NO"].astype(str).str.strip()
        df["LOAD_NO"] = df["LOAD_NO"].astype(str).str.strip()
        
        # Keep physical attribute columns if they exist
        attribute_columns = [
            "长（毫米）", "宽（毫米）", "高（毫米）", "直径（毫米）",
            "FG_PRODUCTION_WT_KG", "ORDER_PIECES", "重量（吨）", "形状"
        ]
        return df

    def _build_history(self) -> HistoricalStats:
        grouped = self.dataframe.groupby("LOAD_NO")["ITEM_NO"].apply(list)
        load_groups = grouped.to_dict()

        combo_counter: Counter = Counter()
        pair_counter: Counter = Counter()
        
        # Extract physical attributes for each ITEM_NO
        item_attributes: Dict[str, Dict[str, Any]] = {}
        attribute_columns = [
            "长（毫米）", "宽（毫米）", "高（毫米）", "直径（毫米）",
            "FG_PRODUCTION_WT_KG", "ORDER_PIECES", "重量（吨）", "形状"
        ]
        
        for item_no in self.dataframe["ITEM_NO"].unique():
            item_data = self.dataframe[self.dataframe["ITEM_NO"] == item_no].iloc[0]
            attrs: Dict[str, Any] = {}
            for col in attribute_columns:
                if col in self.dataframe.columns:
                    value = item_data[col]
                    # Convert to appropriate type
                    if pd.notna(value):
                        if col in ["长（毫米）", "宽（毫米）", "高（毫米）", "直径（毫米）", 
                                   "FG_PRODUCTION_WT_KG", "ORDER_PIECES", "重量（吨）"]:
                            try:
                                attrs[col] = float(value)
                            except (ValueError, TypeError):
                                attrs[col] = value
                        else:
                            attrs[col] = str(value) if pd.notna(value) else None
                    else:
                        attrs[col] = None
            item_attributes[item_no] = attrs

        for items in grouped:
            unique_items = list(dict.fromkeys(items))  # preserve order, drop duplicates
            combo_counter[frozenset(unique_items)] += 1

            for i in range(len(unique_items)):
                for j in range(i + 1, len(unique_items)):
                    pair = frozenset({unique_items[i], unique_items[j]})
                    pair_counter[pair] += 1

        return HistoricalStats(
            load_groups=load_groups,
            combo_frequency=combo_counter,
            pair_frequency=pair_counter,
            item_attributes=item_attributes,
        )

    def _init_openai_client(self, api_key: Optional[str]) -> Optional[OpenAI]:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_KEY
        if not resolved_key:
            return None
        return OpenAI(api_key=resolved_key)

    def _build_context_summary(self) -> str:
        total_loads = len(self.history.load_groups)
        unique_items = self.dataframe["ITEM_NO"].nunique()
        summary_lines = [
            f"Historical loads analysed: {total_loads}",
            f"Unique steel items observed: {unique_items}",
            "",
            "Most frequent loading combinations:",
        ]

        for rank, (combo, count) in enumerate(
            self.history.combo_frequency.most_common(self.top_k_combos), start=1
        ):
            combo_list = sorted(combo)
            summary_lines.append(f"{rank}. {combo_list} -> {count} occurrences")

        if len(summary_lines) == 4:
            summary_lines.append("No repeated combinations found.")

        summary_lines.append("")
        summary_lines.append("Notable pair co-occurrences:")
        for rank, (pair, count) in enumerate(
            self.history.pair_frequency.most_common(self.top_k_combos), start=1
        ):
            pair_list = sorted(pair)
            summary_lines.append(f"{rank}. {pair_list} -> {count} co-loads")

        if len(summary_lines) <= 6:
            summary_lines.append("No frequent pairs detected.")

        # Add physical attribute statistics
        summary_lines.append("")
        summary_lines.append("Physical attribute statistics:")
        attrs = self.history.item_attributes
        if attrs:
            # Calculate statistics for numeric attributes
            numeric_cols = ["长（毫米）", "宽（毫米）", "高（毫米）", "直径（毫米）", 
                           "FG_PRODUCTION_WT_KG", "ORDER_PIECES", "重量（吨）"]
            for col in numeric_cols:
                values = [v.get(col) for v in attrs.values() if v.get(col) is not None]
                if values:
                    try:
                        numeric_vals = [float(v) for v in values]
                        summary_lines.append(
                            f"  {col}: min={min(numeric_vals):.2f}, "
                            f"max={max(numeric_vals):.2f}, "
                            f"avg={sum(numeric_vals)/len(numeric_vals):.2f}"
                        )
                    except (ValueError, TypeError):
                        pass
            
            # Shape distribution
            shapes = [v.get("形状") for v in attrs.values() if v.get("形状")]
            if shapes:
                shape_counter = Counter(shapes)
                summary_lines.append("  Shape distribution:")
                for shape, count in shape_counter.most_common(5):
                    summary_lines.append(f"    {shape}: {count} items")

        return "\n".join(summary_lines)

    def get_context_summary(self) -> str:
        """Public accessor for the textual summary of historical patterns."""
        return self._build_context_summary()

    def plan_with_history(self, new_items: Sequence[str]) -> List[List[str]]:
        """
        Generate a baseline loading plan using the most frequent historical combinations.
        """
        remaining = Counter(str(item).strip() for item in new_items if str(item).strip())
        plan: List[List[str]] = []

        # Sort combos by descending size and frequency to prioritise larger, popular templates.
        sorted_combos = sorted(
            self.history.combo_frequency.items(),
            key=lambda kv: (-len(kv[0]), -kv[1]),
        )

        for combo, _ in sorted_combos:
            combo_list = list(combo)
            if not combo_list:
                continue
            if all(remaining[item] > 0 for item in combo_list):
                assigned = []
                for item in combo_list:
                    if remaining[item] > 0:
                        remaining[item] -= 1
                        assigned.append(item)
                if assigned:
                    plan.append(assigned)

        # Assign any leftover items individually.
        for item, count in list(remaining.items()):
            for _ in range(count):
                plan.append([item])

        return plan

    def plan_with_openai(
        self,
        new_items: Sequence[str],
        item_attributes: Optional[Dict[str, Dict[str, Any]]] = None,
        temperature: float = 0.1,
        use_history_plan: bool = True,
    ) -> Dict[str, Any]:
        """
        Ask an OpenAI model to propose a loading plan for the new items.

        Args:
            new_items: List of ITEM_NO identifiers
            item_attributes: Optional dict mapping ITEM_NO to physical attributes
                (length, width, height, diameter, weight, pieces, shape, etc.)
            temperature: Generation temperature
            use_history_plan: Whether to include baseline plan

        Returns a dictionary with keys:
        - "baseline_plan": plan generated from history (optional)
        - "response_text": natural-language recommendation from the model
        """
        if not self.client:
            raise RuntimeError(
                "OpenAI client is not initialised. Set the OPENAI_API_KEY environment variable "
                "or pass the api key explicitly when constructing SteelLoadingPlanner."
            )

        baseline_plan: Optional[List[List[str]]] = None
        if use_history_plan:
            baseline_plan = self.plan_with_history(new_items)

        context_summary = self._build_context_summary()
        new_items_list = [str(item).strip() for item in new_items if str(item).strip()]

        system_prompt = (
            "You are a logistics planner specialising in loading plans for steel transport. "
            "Consider physical dimensions (length, width, height, diameter), weight, quantity, "
            "and shape when making loading decisions. Leverage historical patterns to recommend "
            "efficient groupings while keeping practical constraints in mind. "
            "Answer in concise natural language, clearly listing each truck and its assigned steel items."
        )

        user_prompt = (
            f"Historical summary:\n{context_summary}\n\n"
            f"New steel items to load (ITEM_NO): {new_items_list}\n"
        )

        # Add physical attributes for new items
        if item_attributes:
            user_prompt += "\nPhysical attributes of new items:\n"
            for item_no in new_items_list:
                attrs = item_attributes.get(item_no, {})
                if attrs:
                    attr_lines = [f"  {item_no}:"]
                    for key, value in attrs.items():
                        if value is not None:
                            attr_lines.append(f"    {key}: {value}")
                    user_prompt += "\n".join(attr_lines) + "\n"
                else:
                    # Try to get from historical data if available
                    hist_attrs = self.history.item_attributes.get(item_no, {})
                    if hist_attrs:
                        attr_lines = [f"  {item_no} (from history):"]
                        for key, value in hist_attrs.items():
                            if value is not None:
                                attr_lines.append(f"    {key}: {value}")
                        user_prompt += "\n".join(attr_lines) + "\n"
        else:
            # Try to get attributes from historical data
            user_prompt += "\nPhysical attributes (from historical data if available):\n"
            for item_no in new_items_list:
                attrs = self.history.item_attributes.get(item_no, {})
                if attrs:
                    attr_lines = [f"  {item_no}:"]
                    for key, value in attrs.items():
                        if value is not None:
                            attr_lines.append(f"    {key}: {value}")
                    user_prompt += "\n".join(attr_lines) + "\n"

        if baseline_plan:
            user_prompt += (
                "\nBaseline plan from historical templates:\n"
                f"{baseline_plan}\n"
                "You may refine this plan if needed. "
            )

        user_prompt += (
            "\nPlease provide a loading plan considering:\n"
            "- Physical dimensions and weight constraints\n"
            "- Shape compatibility\n"
            "- Historical loading patterns\n"
            "- Efficient space utilization"
        )

        response_text = self._call_openai(system_prompt, user_prompt, temperature)

        return {
            "baseline_plan": baseline_plan or [],
            "response_text": response_text,
        }

    def _call_openai(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return response.output[0].content[0].text.strip()  # type: ignore[attr-defined]
        except AttributeError:
            # Fall back to chat completions for older client versions.
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return completion.choices[0].message.content.strip()  # type: ignore[attr-defined]

def _normalize_item_input(raw_input: str) -> List[str]:
    """Parse user input that may contain commas, whitespace, or new lines."""
    import re

    items: List[str] = []
    for line in raw_input.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [token.strip() for token in re.split(r"[,\s]+", line) if token.strip()]
        items.extend(parts)
    return items


def _plan_to_dataframe(plan: Sequence[Sequence[str]]) -> pd.DataFrame:
    if not plan:
        return pd.DataFrame(columns=["Load #", "Items"])
    return pd.DataFrame(
        {
            "Load #": [idx + 1 for idx, _ in enumerate(plan)],
            "Items": [", ".join(load) for load in plan],
        }
    )


def run_streamlit_app() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Streamlit is required to run the interactive interface. "
            "Install it with `pip install streamlit`."
        ) from exc

    st.set_page_config(page_title="Steel Loading Planner", layout="wide")
    st.title("Steel Loading Planner")
    st.write(
        "Use historical loading records and OpenAI-assisted reasoning to craft loading plans "
        "for new steel shipments."
    )

    with st.sidebar:
        st.header("Configuration")
        data_path_input = st.text_input("Historical data file", "data.xlsx")
        resolved_data_path = data_path_input.strip() or "data.xlsx"

        default_key = os.getenv("OPENAI_API_KEY", DEFAULT_OPENAI_KEY)
        openai_key_input = st.text_input(
            "OpenAI API key",
            value=default_key,
            type="password",
            help="The key is used client-side only for this session.",
        )

        model_name_input = st.text_input("OpenAI model", "gpt-4o-mini")
        temperature = st.slider(
            "Generation temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05
        )
        use_history_plan = st.checkbox(
            "Provide historical baseline to OpenAI", value=True,
            help="If enabled, the baseline plan generated from historical patterns will be passed "
                 "to the model as a starting point.",
        )
        top_k = st.slider(
            "Number of frequent combinations to summarise",
            min_value=3,
            max_value=30,
            value=10,
        )
        show_history = st.checkbox("Show historical summary", value=True)

    planner: Optional[SteelLoadingPlanner] = None
    planner_error: Optional[str] = None
    data_file = Path(resolved_data_path)

    if not data_file.exists():
        planner_error = f"Data file '{data_file}' not found."
    else:
        try:
            planner = SteelLoadingPlanner(
                data_path=str(data_file),
                openai_api_key=openai_key_input.strip() or None,
                model=model_name_input.strip() or "gpt-4o-mini",
                top_k_combos=top_k,
            )
        except Exception as exc:  # pylint: disable=broad-except
            planner_error = str(exc)

    st.subheader("New Load Request")
    st.write(
        "Enter the identifiers of steel items that require loading. "
        "You can optionally specify physical attributes for each item. "
        "If attributes are not provided, the system will try to retrieve them from historical data."
    )

    new_items_input = st.text_area(
        "Steel item identifiers (ITEM_NO)",
        placeholder="ITEM_001\nITEM_002\nITEM_003",
        height=120,
        key="items_input",
    )
    
    parsed_items = _normalize_item_input(new_items_input)
    item_attributes: Optional[Dict[str, Dict[str, Any]]] = None
    
    enable_attributes = st.checkbox(
        "Specify physical attributes manually",
        value=False,
        help="If checked, you can input physical attributes for each item. "
             "Otherwise, attributes will be retrieved from historical data if available."
    )
    
    if enable_attributes and parsed_items:
        st.write("**Physical Attributes (optional, leave blank to use historical data):**")
        with st.expander("Edit attributes", expanded=True):
            for idx, item_no in enumerate(parsed_items):
                st.write(f"**{item_no}**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    length = st.number_input(
                        "Length (mm)",
                        min_value=0.0,
                        value=None,
                        key=f"length_{idx}",
                    )
                with col2:
                    width = st.number_input(
                        "Width (mm)",
                        min_value=0.0,
                        value=None,
                        key=f"width_{idx}",
                    )
                with col3:
                    height = st.number_input(
                        "Height (mm)",
                        min_value=0.0,
                        value=None,
                        key=f"height_{idx}",
                    )
                with col4:
                    diameter = st.number_input(
                        "Diameter (mm)",
                        min_value=0.0,
                        value=None,
                        key=f"diameter_{idx}",
                    )
                
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    weight_kg = st.number_input(
                        "Weight (kg)",
                        min_value=0.0,
                        value=None,
                        key=f"weight_kg_{idx}",
                    )
                with col6:
                    pieces = st.number_input(
                        "Pieces",
                        min_value=0,
                        value=None,
                        key=f"pieces_{idx}",
                    )
                with col7:
                    weight_ton = st.number_input(
                        "Weight (ton)",
                        min_value=0.0,
                        value=None,
                        key=f"weight_ton_{idx}",
                    )
                with col8:
                    shape = st.text_input(
                        "Shape",
                        value="",
                        key=f"shape_{idx}",
                    )
                
                attrs: Dict[str, Any] = {}
                if length is not None:
                    attrs["长（毫米）"] = length
                if width is not None:
                    attrs["宽（毫米）"] = width
                if height is not None:
                    attrs["高（毫米）"] = height
                if diameter is not None:
                    attrs["直径（毫米）"] = diameter
                if weight_kg is not None:
                    attrs["FG_PRODUCTION_WT_KG"] = weight_kg
                if pieces is not None:
                    attrs["ORDER_PIECES"] = int(pieces)
                if weight_ton is not None:
                    attrs["重量（吨）"] = weight_ton
                if shape:
                    attrs["形状"] = shape
                
                if attrs:
                    if item_attributes is None:
                        item_attributes = {}
                    item_attributes[item_no] = attrs
                st.divider()

    openai_result: Optional[Dict[str, Any]] = None
    
    if st.button("Generate plan with OpenAI", type="primary"):
        if planner is None:
            st.error(planner_error or "Planner could not be initialised.")
        elif not parsed_items:
            st.warning("Please provide at least one steel item.")
        else:
            try:
                openai_result = planner.plan_with_openai(
                    parsed_items,
                    item_attributes=item_attributes,
                    temperature=temperature,
                    use_history_plan=use_history_plan,
                )
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Failed to retrieve response from OpenAI: {exc}")

    if openai_result is not None:
        st.subheader("Plan suggested by OpenAI")
        response_text = openai_result.get("response_text", "")
        if response_text:
            st.markdown(response_text)
        else:
            st.warning("The model did not return any content.")

        if openai_result.get("baseline_plan"):
            with st.expander("Baseline provided to the model"):
                st.dataframe(
                    _plan_to_dataframe(openai_result["baseline_plan"]),
                    use_container_width=True,
                )

        st.download_button(
            "Download OpenAI response",
            data=json.dumps(openai_result, indent=2, ensure_ascii=False),
            file_name="loading_plan.json",
            mime="application/json",
        )

    if planner_error and planner is None:
        st.error(planner_error)

    if planner and show_history:
        st.subheader("Historical insights")
        st.code(planner.get_context_summary())


def demo() -> None:
    """
    Example entry point.
    Set the environment variable OPENAI_API_KEY before running.
    """
    planner = SteelLoadingPlanner(
        data_path="data.xlsx",
        openai_api_key=os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_KEY,
    )

    # Example: replace with actual incoming steel item IDs.
    new_items = ["ITEM_101", "ITEM_202", "ITEM_303", "ITEM_404"]

    print("Baseline plan based on historical data:")
    print(planner.plan_with_history(new_items))

    if planner.client:
        print("\nPlan generated with OpenAI:")
        result = planner.plan_with_openai(new_items)
        print(result.get("response_text", "No response received."))
    else:
        print("\nOpenAI API key not provided. Skipping model-generated plan.")


def running_in_streamlit() -> bool:
    """Best-effort check to see if the script is being executed inside Streamlit."""
    return any(module_name.startswith("streamlit") for module_name in sys.modules)


if __name__ == "__main__":
    if running_in_streamlit():
        run_streamlit_app()
    else:
        demo()

