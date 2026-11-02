#!/usr/bin/env python3
"""Run controlled-insertion tests using ONLY 10 real Pile hosts and real BBC news passages.
Every test produces one self-contained JSON file in the same folder.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import traceback
import tempfile
import math
import platform
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import controlled_insertion as engine

PILE = "EleutherAI/the_pile_deduplicated"
BBC = "SetFit/bbc-news"


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def extract_realtime_sources():
    from datasets import load_dataset
    # Use exactly the same production function used by the insertion engine.
    hosts = engine.load_hosts({"name": PILE, "split": "train", "take": {"mode": "first_n", "n": 10}})
    news = load_dataset(BBC, split="train", streaming=True)
    # BBC rows are real; preserve row index and full original text for provenance.
    bbc_rows = []
    for row_id, row in enumerate(news):
        txt = row.get("text")
        if isinstance(txt, str) and len(txt.strip()) >= 120 and isinstance(row.get("label_text"), str):
            bbc_rows.append({"row": row_id, "text": txt, "label": row.get("label_text")})
        if len(bbc_rows) == 4:
            break
    assert len(hosts) == 10 and len(bbc_rows) == 4
    assert [h["source_id"] for h in hosts] == list(range(10))
    return hosts, bbc_rows


def t(id_, span, selection=None, position=None, overrides=None, *, source=None):
    entry = {"id": id_, "source": source or {"mode": "literal", "text": span},
             "selection": selection or {"mode": "explicit", "source_ids": [0]},
             "position": position or {"mode": "beginning"}}
    if overrides is not None:
        entry["per_document_positions"] = overrides
    return entry


def config(targets, separator="\n\n", seed=42, sample_external=None):
    return {"seed": seed, "separator": separator, "targets": targets, "sample_external": sample_external}


def run_checks(hosts, docs, infos, recs, tokenizer, expected):
    """Independent reconstruction and metadata verification; no trust in engine checks."""
    checks = []
    assert len(docs) == len(hosts) == 10
    assert [d["pile_source_id"] for d in docs] == [h["source_id"] for h in hosts]
    assert len(recs) == sum(d["insertion_count"] for d in docs)
    assert len(recs) == sum(info["frequency"] for info in infos)
    checks.append("10 documents and source IDs retained in original order")
    for i, (host, doc) in enumerate(zip(hosts, docs)):
        original = host["text"]
        modified = doc["text"]
        assert doc["output_document_index"] == i
        assert doc["original_sha256"] == hashlib.sha256(original.encode("utf-8")).hexdigest()
        assert doc["modified_sha256"] == hashlib.sha256(modified.encode("utf-8")).hexdigest()
        own = [r for r in recs if r["output_document_index"] == i]
        assert doc["insertion_count"] == len(own)
        if not own:
            assert modified == original, f"Unexpected modification to document {i}"
        # Carefully remove exactly the annotated added text and separators (right to left).
        for r in sorted(own, key=lambda item: item["inserted_start_char"], reverse=True):
            a, b = r["inserted_start_char"], r["inserted_end_char"]
            ls, rs = r["left_separator"], r["right_separator"]
            assert modified[a:b] == r["inserted_span"]
            assert modified[a-len(ls):a] == ls if ls else True
            assert modified[b:b+len(rs)] == rs if rs else True
            modified = modified[:a-len(ls)] + modified[b+len(rs):]
            assert 0 <= r["original_character_offset"] <= len(original)
            obs = r["final_token_observation"]
            actual_offsets, _ = engine.token_map(doc["text"], tokenizer)
            overlap = [j for j, (start, stop) in enumerate(actual_offsets) if start < b and stop > a]
            interval = [overlap[0], overlap[-1]+1] if overlap else None
            assert obs["modified_token_count"] == len(actual_offsets)
            assert obs["span_overlapping_token_interval"] == interval
            crossing = [j for j in overlap if actual_offsets[j][0] < a or actual_offsets[j][1] > b]
            assert obs["boundary_crossing_token_indices"] == crossing
            assert obs["no_overlapping_tokens"] == (not overlap)
            assert obs["span_first_token_fraction"] == (overlap[0] / len(actual_offsets) if overlap else None)
            assert r["original_token_count"] == len(engine.token_map(original, tokenizer)[0])
        assert modified == original, f"Reconstruction mismatch on document {i}"
        # Independent forward construction from original offsets. This also checks
        # ordering and exact separators for multiple targets at the same offset.
        cursor = 0
        expected_parts = []
        for r in sorted(own, key=lambda x: (x["original_character_offset"], x["inserted_start_char"])):
            at = r["original_character_offset"]
            assert at >= cursor
            expected_parts.extend([original[cursor:at], r["left_separator"], r["inserted_span"], r["right_separator"]])
            cursor = at
        expected_parts.append(original[cursor:])
        assert doc["text"] == "".join(expected_parts), f"Exact output mismatch on document {i}"

    checks.append("every original byte/character reconstructs exactly; unchanged hosts unchanged")
    checks.append("inserted span positions, separators, document hashes and counts match")
    checks.append("final retokenization metadata independently verified with selected fast tokenizer")
    for info in infos:
        selected = [r for r in recs if r["target_id"] == info["id"]]
        assert len(selected) == info["frequency"]
        assert sorted(r["pile_source_id"] for r in selected) == sorted(info["host_source_ids"])
        assert len(set(r["pile_source_id"] for r in selected)) == len(selected)
    if "kind" in expected and expected["kind"] != "engine":
        pass
    if "count" in expected:
        assert len(recs) == expected["count"]
        checks.append("exact expected insertion frequency")
    if "ids" in expected:
        assert infos[0]["host_source_ids"] == expected["ids"]
        checks.append("exact selected source IDs")
    if "allowed" in expected:
        assert set(infos[0]["host_source_ids"]).issubset(expected["allowed"])
        checks.append("source IDs restricted to selected dataset region")
    if "offset" in expected:
        assert recs[0]["original_character_offset"] == expected["offset"]
        checks.append("exact original character insertion offset")
    if "position_fraction" in expected:
        r = recs[0]
        assert r["original_character_offset"] == round(len(hosts[0]["text"]) * expected["position_fraction"])
        checks.append("exact character-fraction rounding")
    if "token_index" in expected:
        assert recs[0]["position"]["requested_token_boundary"] == expected["token_index"]
        checks.append("exact requested token index")
    if "delimiter" in expected:
        assert recs[0]["position"]["matched_delimiter"] == expected["delimiter"]
        actual = recs[0]["original_character_offset"]
        assert hosts[0]["text"][:actual].endswith(expected["delimiter"])
        checks.append("insertion is immediately after the requested delimiter")
    if "fallback" in expected:
        assert recs[0]["position"]["delimiter_fallback"] == expected["fallback"]
        checks.append("correct missing-delimiter fallback")
    if "span" in expected:
        assert recs[0]["inserted_span"] == expected["span"]
        checks.append("extracted BBC target matches expected exact substring")
    if "span_length" in expected:
        assert len(recs[0]["inserted_span"]) == expected["span_length"]
        checks.append("BBC extracted passage has correct character length")
    if "offsets_by_id" in expected:
        actual = {r["pile_source_id"]:r["original_character_offset"] for r in recs}
        assert actual == expected["offsets_by_id"], (actual, expected["offsets_by_id"])
        checks.append("per-document exact original insertion offsets")
    if "target_order" in expected:
        actual = [r["target_id"] for r in sorted(recs, key=lambda r:r["inserted_start_char"])]
        assert actual == expected["target_order"], (actual, expected["target_order"])
        checks.append("exact target ordering at shared insertion boundary")
    if "separator" in expected:
        for r in recs:
            expected_left = expected["separator"] if r["original_character_offset"] > 0 else ""
            expected_right = expected["separator"] if r["original_character_offset"] < len(hosts[r["output_document_index"]]["text"]) else ""
            assert r["left_separator"] == expected_left
            assert r["right_separator"] == expected_right
        checks.append("exact separator placement")
    if "fallback" in expected and recs:
        if expected["fallback"] == "end":
            assert recs[0]["original_character_offset"] == len(hosts[0]["text"])
        if expected["fallback"] == "requested":
            assert recs[0]["original_character_offset"] == recs[0]["position"]["requested_character_offset"]
    if "repeat" in expected:
        docs2, infos2, recs2 = engine.insert_targets(copy.deepcopy(expected["config"]), copy.deepcopy(hosts), tokenizer)
        assert (docs, infos, recs) == (docs2, infos2, recs2)
        checks.append("same random seed reproduces complete result")
    return checks


def build_cases(hosts, bbc, tokenizer):
    # Source text is *always* taken from BBC Hugging Face rows.
    # Use one fixed actual passage for positional/frequency comparisons.
    passage = bbc[0]["text"][20:100]
    passage_b = bbc[1]["text"][40:135]
    passage_c = bbc[2]["text"][15:110]
    sources = [passage, passage_b, passage_c]
    # Some source snippets might already occur in Pile; choose a unique BBC substring if necessary.
    for k, span in enumerate(sources):
        if not span or any(span in h["text"] for h in hosts):
            raise RuntimeError(f"BBC passage {k} is empty or already occurs in a Pile host")
    C = []
    def add(name, desc, cfg, **expected):
        C.append({"name":name, "description":desc, "configuration":cfg, "expected":expected})
    base = lambda pos=None, sel=None, overrides=None: config([t("BBC", passage, sel, pos, overrides)])
    for mode, offset in (("beginning",0),("middle",len(hosts[0]["text"])//2),("end",len(hosts[0]["text"]))):
        add("position_"+mode, "BBC span at original "+mode, base({"mode":mode}), offset=offset, count=1)
    for n in (0, 1, 7, min(50,len(hosts[0]["text"]))):
        add("character_offset_"+str(n), "Exact offset "+str(n), base({"mode":"character_offset","value":n}), offset=n)
    for f in (0,.1,.25,.5,.75,.9,1):
        suffix=str(f).replace('.','_')
        add("character_fraction_"+suffix,"Original character fraction",base({"mode":"character_fraction","fraction":f}),position_fraction=f)
        add("percentage_"+suffix,"Percentage alias",base({"mode":"percentage","value":f}),position_fraction=f)
    add("random_character","Random character position with fixed seed",base({"mode":"random_character"}),repeat=True)
    offsets, boundaries = engine.token_map(hosts[0]["text"],tokenizer)
    good = [j for j,b in enumerate(boundaries) if b is not None]
    for n in sorted({0, min(1,len(offsets)), min(5,len(offsets)), len(offsets)}):
        if n in good:
            add("token_offset_"+str(n),"Original tokenizer token boundary",base({"mode":"token_offset","value":n}),offset=boundaries[n],token_index=n)
    for frac in (0,.25,.5,.75,1):
        for mode in ("token_fraction","token_percentage"):
            idx=round(len(offsets)*frac)
            if boundaries[idx] is not None:
                add(mode+"_"+str(frac).replace('.','_'),"Original tokenizer token fraction",base({"mode":mode,"fraction":frac}),offset=boundaries[idx],token_index=idx)
    add("random_token","Random valid original tokenizer token boundary",base({"mode":"random_token"}),repeat=True)
    # Dynamically choose real delimiters actually occurring in Pile document 0.
    for d, label in ((".","period"),("\n","newline"),(" ","space")):
        if d in hosts[0]["text"]:
            add("delimiter_character_"+label,"Align to real delimiter in Pile text",base({"mode":"character_fraction_delimiter","fraction":0,"delimiters":[d]}),delimiter=d)
            valid = {b for b in boundaries if b is not None}
            if any(i+len(d) in valid for i in range(len(hosts[0]["text"])) if hosts[0]["text"].startswith(d,i)):
                add("delimiter_token_"+label,"Align to eligible GPT-2 delimiter boundary",base({"mode":"token_fraction_delimiter","fraction":0,"delimiters":[d]}),delimiter=d)
    for mode in ("character_fraction_delimiter","token_fraction_delimiter"):
        for policy in ("requested","end","error"):
            cfg=base({"mode":mode,"fraction":.3,"delimiters":["<UNMATCHED_DELIMITER_7c91>"],"on_missing":policy})
            add("delimiter_"+mode+"_missing_"+policy,"Real document with missing delimiter and fallback",cfg, **({"error":"ValueError"} if policy=="error" else {"fallback":policy}))
    add("frequency_explicit","Exactly these Pile source IDs",base(sel={"mode":"explicit","source_ids":[0,4,9]}),ids=[0,4,9],count=3)
    for count in (0,1,3,5,10):
        add("frequency_random_"+str(count),"Exact count of distinct real Pile hosts",base(sel={"mode":"random","frequency":count,"stage":"all"}),count=count,repeat=True)
    add("frequency_range","Three hosts drawn only from loaded indices [2,7)",base(sel={"mode":"range","frequency":3,"start_output_index":2,"stop_output_index":7}),count=3,allowed=list(range(2,7)))
    add("frequency_every_n","Every third loaded document",base(sel={"mode":"every_n","frequency":4,"every_n":3}),count=4,ids=[0,3,6,9])
    for stage, allowed in (("early",[0,1,2]),("middle",[3,4,5]),("late",[6,7,8,9]),("all",list(range(10)))):
        add("region_"+stage,"Real Pile stage "+stage,base(sel={"mode":"random","frequency":2,"stage":stage}),allowed=allowed,count=2)
    add("region_custom","20% to 80% region of the 10 loaded Pile docs",base(sel={"mode":"random","frequency":3,"stage":"custom","start_fraction":.2,"end_fraction":.8}),allowed=list(range(2,8)),count=3)
    add("per_document_overrides","BBC target positioned differently in Pile docs 0,1,2",base(sel={"mode":"explicit","source_ids":[0,1,2]},overrides={"0":{"mode":"beginning"},"1":{"mode":"middle"},"2":{"mode":"end"}}),count=3,offsets_by_id={0:0,1:len(hosts[1]["text"])//2,2:len(hosts[2]["text"])})
    for label, target_list in (
        ("same_host",[t("BBC_A",passage),t("BBC_B",passage_b,position={"mode":"end"})]),
        ("same_offset",[t("BBC_A",passage),t("BBC_B",passage_b)]),
        ("different_hosts",[t("BBC_A",passage),t("BBC_B",passage_b,selection={"mode":"explicit","source_ids":[9]})]),
    ):
        add("multiple_targets_"+label,"Two real BBC passages and preserved original text",config(target_list),count=2, **({"target_order":["BBC_A","BBC_B"]} if label=="same_offset" else {}))
    for label, sep in (("default","\n\n"),("custom"," | BBC | "),("empty","")):
        add("separator_"+label,"Insertion with explicit separator style",base() if label=="default" else config([t("BBC",passage)],separator=sep),count=1,separator=sep)
    # Real Hugging Face target extraction (get_span uses the actual BBC dataset by name).
    source = lambda spec: {"mode":"hf","dataset":BBC,"split":"train","index":bbc[0]["row"],"fields":"text","extract":spec}
    add("bbc_source_full","Whole article from real BBC dataset",config([t("BBC_FULL",None,source=source({"mode":"all"}))]),span=bbc[0]["text"])
    add("bbc_source_characters","BBC article character slice",config([t("BBC_CHARS",None,source=source({"mode":"characters","start":20,"length":80}))]),span=passage)
    add("bbc_source_two_fields","Combine BBC text and BBC label_text from the same real row",config([t("BBC_TWO_FIELDS",None,source={"mode":"hf","dataset":BBC,"split":"train","index":bbc[0]["row"],"fields":["text","label_text"],"join_with":" | ","extract":{"mode":"all"}})]),span=bbc[0]["text"]+" | "+bbc[0]["label"])
    add("bbc_source_random_characters","Seeded BBC article character passage",config([t("BBC_RANDOM_CHARS",None,source=source({"mode":"random_characters","length":80}))]),span_length=80,repeat=True)
    news_offsets, news_boundaries = engine.token_map(bbc[0]["text"],tokenizer)
    valid_start=next((j for j in range(len(news_offsets)-10) if news_boundaries[j] is not None and news_boundaries[j+10] is not None),None)
    if valid_start is not None:
        add("bbc_source_tokens","BBC article original tokenizer token passage",config([t("BBC_TOKENS",None,source=source({"mode":"tokens","start":valid_start,"length":10}))]),span=bbc[0]["text"][news_boundaries[valid_start]:news_boundaries[valid_start+10]])
        add("bbc_source_random_tokens","Seeded real BBC original-token passage",config([t("BBC_RANDOM_TOKENS",None,source=source({"mode":"random_tokens","length":10}))]),repeat=True)
    add("bbc_sampling_multiple","Two unique real BBC articles with independent target IDs",config([],sample_external={"dataset":BBC,"split":"train","fields":"text","count":2,"pool_size":4,"extract":{"mode":"characters","start":30,"length":85},"selection":{"mode":"random","frequency":1,"stage":"all"},"position":{"mode":"beginning"},"per_target":{"0":{"selection":{"mode":"explicit","source_ids":[0,9]}}}}),count=3)
    # All these error tests still use real hosts and BBC span; errors are EXPECTED.
    invalids = {
      "duplicate_host_ids":base(sel={"mode":"explicit","source_ids":[0,0]}),
      "unloaded_host_id":base(sel={"mode":"explicit","source_ids":[11]}),
      "frequency_excess":base(sel={"mode":"random","frequency":11}),
      "negative_frequency":base(sel={"mode":"random","frequency":-1}),
      "character_offset_excess":base(pos={"mode":"character_offset","value":len(hosts[0]["text"])+1}),
      "negative_fraction":base(pos={"mode":"character_fraction","fraction":-.1}),
      "fraction_excess":base(pos={"mode":"character_fraction","fraction":1.1}),
      "token_offset_excess":base(pos={"mode":"token_offset","value":len(offsets)+1}),
      "invalid_delimiters":base(pos={"mode":"character_fraction_delimiter","fraction":.5,"delimiters":[]}),
      "duplicate_target_ids":config([t("same",passage),t("same",passage_b)]),
      "contained_target_text":config([t("a",passage),t("b",passage[:30])]),
      "invalid_separator":config([t("BBC",passage)],separator=99),
      "invalid_stage":base(sel={"mode":"random","frequency":1,"stage":"unknown"}),
      "invalid_position_mode":base(pos={"mode":"unknown"}),
      "invalid_custom_region":base(sel={"mode":"random","frequency":1,"stage":"custom","start_fraction":.9,"end_fraction":.1}),
      "insufficient_stage_hosts":base(sel={"mode":"random","frequency":4,"stage":"early"}),
      "invalid_every_n_step":base(sel={"mode":"every_n","frequency":1,"every_n":0}),
      "empty_literal_source":config([t("empty",None,source={"mode":"literal","text":""})]),
    }
    for label,cfg in invalids.items():
        add("reject_"+label,"Invalid request should be rejected on real inputs",cfg,error="ValueError")
    # Loading modes are each verified against the same real first-10 snapshot.
    for label, take, ids in (
        ("first_n", {"mode":"first_n","n":10}, list(range(10))),
        ("source_ids", {"mode":"source_ids","ids":[9,0,4]}, [0,4,9]),
        ("source_range", {"mode":"source_range","start":3,"stop":7}, [3,4,5,6]),
    ):
        add("load_hosts_"+label, "Real Pile streaming host loading mode", config([]), kind="load", take=take, loaded_ids=ids)
    add("end_to_end_write_outputs", "Execute real run() and read its three persisted output files", base(), kind="run")
    add("reject_output_overwrite", "Production run() must not overwrite output folder", base(), kind="overwrite")
    add("token_map_without_tokenizer", "Token mode rejects missing tokenizer", base({"mode":"token_offset","value":1}), kind="no_tokenizer", error="ValueError")
    add("token_map_empty_real_substring", "Empty substring from real Pile text gives no token boundaries", config([]), kind="token_map_empty")
    add("reject_second_insertion_same_span", "Attempt to insert BBC passage twice into a derived real Pile host", base(), kind="repeat_reject")
    add("no_targets", "All 10 real Pile documents unchanged without targets", config([]), count=0)
    add("empty_explicit_selection", "No hosts explicitly selected", base(sel={"mode":"explicit","source_ids":[]}), count=0)
    add("every_n_nonzero_start", "Every second host starting at index 1", base(sel={"mode":"every_n","every_n":2,"frequency":3,"start_output_index":1}), ids=[1,3,5], count=3)
    add("random_range_zero", "Zero random host selections in range", base(sel={"mode":"range","frequency":0,"start_output_index":2,"stop_output_index":6}), count=0)
    add("multiple_targets_three_same_host", "Three distinct BBC passages on same Pile host", config([t("A",passage),t("B",passage_b,position={"mode":"middle"}),t("C",passage_c,position={"mode":"end"})]), count=3, target_order=["A","B","C"])
    add("multiple_targets_three_same_offset", "Three BBC passages at same offset", config([t("A",passage),t("B",passage_b),t("C",passage_c)]), count=3, target_order=["A","B","C"])
    existing_delims = [d for d in ("\n", ".", " ") if d in hosts[0]["text"]]
    if len(existing_delims) >= 2:
        earliest = min((hosts[0]["text"].find(d), idx, d) for idx, d in enumerate(existing_delims))
        add("custom_delimiter_order", "Several delimiters; earliest starting match must win", base({"mode":"character_fraction_delimiter","fraction":0,"delimiters":existing_delims}), count=1, delimiter=earliest[2], offset=earliest[0]+len(earliest[2]))
    invalid_extra = {
       "negative_character_offset": base(pos={"mode":"character_offset","value":-1}),
       "negative_token_offset": base(pos={"mode":"token_offset","value":-1}),
       "nan_fraction": base(pos={"mode":"character_fraction","fraction":float("nan")}),
       "infinite_fraction": base(pos={"mode":"character_fraction","fraction":float("inf")}),
       "wrong_fraction_type": base(pos={"mode":"character_fraction","fraction":"0.5"}),
       "invalid_range_reverse": base(sel={"mode":"range","frequency":1,"start_output_index":8,"stop_output_index":2}),
       "range_overflow": base(sel={"mode":"range","frequency":1,"start_output_index":2,"stop_output_index":12}),
       "every_n_start_overflow": base(sel={"mode":"every_n","every_n":2,"frequency":1,"start_output_index":10}),
       "every_n_excess": base(sel={"mode":"every_n","every_n":4,"frequency":5}),
       "invalid_missing_delimiter_policy": base(pos={"mode":"character_fraction_delimiter","fraction":0,"delimiters":["<NEVER_MATCH_98f7>"],"on_missing":"invalid"}),
       "empty_string_delimiter": base(pos={"mode":"character_fraction_delimiter","fraction":0,"delimiters":[""]}),
       "duplicate_target_text": config([t("A",passage),t("B",passage)]),
       "no_loaded_hosts": config([t("A",passage)]),
    }
    for label,cfg in invalid_extra.items():
        add("reject_"+label, "Additional boundary/invalid-input verification", cfg, kind="empty_hosts" if label=="no_loaded_hosts" else "engine", error="ValueError")
    return C, sources


def execute(case, hosts, tokenizer, out, provenance):
    cfg=case["configuration"]
    expected=case["expected"]
    result={"case":case["name"],"description":case["description"],"status":"FAIL","expected":{k:v for k,v in expected.items() if k!="config"},"checks":[],"error":None}
    docs=infos=recs=None
    try:
        kind = expected.get("kind", "engine")
        if kind == "load":
            loaded = engine.load_hosts({"name":PILE,"split":"train","take":expected["take"]})
            wanted = expected["loaded_ids"]
            assert [h["source_id"] for h in loaded] == wanted
            assert loaded == [hosts[i] for i in wanted]
            docs,infos,recs = [],[],[]
            checks = ["correct subset of real Pile host records, source IDs and original ordering"]
            result.update(status="PASS",checks=checks,loaded_hosts=loaded)
        elif kind == "run":
            with tempfile.TemporaryDirectory(prefix="pile_insert_run_") as d:
                full_cfg = copy.deepcopy(cfg)
                full_cfg["output_dir"] = str(Path(d)/"experiment")
                full_cfg["pile"] = {"name":PILE,"split":"train","take":{"mode":"first_n","n":10}}
                engine.run(full_cfg, tokenizer=tokenizer)
                folder = Path(full_cfg["output_dir"])
                original_lines = [json.loads(line) for line in (folder/"original_hosts.jsonl").read_text(encoding="utf-8").splitlines()]
                modified_lines = [json.loads(line) for line in (folder/"modified_dataset.jsonl").read_text(encoding="utf-8").splitlines()]
                metadata = json.loads((folder/"insertion_metadata.json").read_text(encoding="utf-8"))
                assert original_lines == hosts
                assert len(modified_lines) == 10 and metadata["total_documents"] == 10
                assert metadata["total_insertions"] == len(metadata["insertions"])
                docs,infos,recs = modified_lines,metadata["targets"],metadata["insertions"]
                checks = run_checks(hosts,docs,infos,recs,tokenizer,{"count":1})
                checks.append("production run() generated three valid files and matching metadata")
                result.update(status="PASS",checks=checks,production_files={"original_hosts.jsonl":original_lines,"modified_dataset.jsonl":modified_lines,"insertion_metadata.json":metadata})
        elif kind == "overwrite":
            with tempfile.TemporaryDirectory(prefix="pile_nooverwrite_") as d:
                folder = Path(d)/"existing"
                folder.mkdir()
                marker = folder/"sentinel.txt"
                marker.write_text("must survive", encoding="utf-8")
                full_cfg=copy.deepcopy(cfg)
                full_cfg.update(output_dir=str(folder), pile={"name":PILE,"split":"train","take":{"mode":"first_n","n":10}})
                try:
                    engine.run(full_cfg, tokenizer=tokenizer)
                except FileExistsError:
                    assert marker.read_text(encoding="utf-8") == "must survive"
                    result.update(status="PASS", checks=["existing experiment folder remains unmodified"])
                else:
                    raise AssertionError("Engine overwrote existing output directory")
        elif kind == "no_tokenizer":
            try:
                engine.insert_targets(copy.deepcopy(cfg),copy.deepcopy(hosts),None)
            except ValueError as e:
                assert "fast tokenizer" in str(e).lower()
                result.update(status="PASS",checks=["token mode correctly rejects missing tokenizer"],error={"expected":True,"type":"ValueError","message":str(e)})
            else:
                raise AssertionError("Token insertion accepted missing tokenizer")
        elif kind == "token_map_empty":
            empty = hosts[0]["text"][:0]
            offsets, boundaries = engine.token_map(empty, tokenizer)
            assert offsets == [] and boundaries == [0]
            result.update(status="PASS",checks=["empty substring has correct token offset mapping"])
        elif kind == "repeat_reject":
            first_docs, _, _ = engine.insert_targets(copy.deepcopy(cfg),copy.deepcopy(hosts),tokenizer)
            edited_hosts = [{"source_id":d["pile_source_id"],"text":d["text"]} for d in first_docs]
            # The second request must reject the already-inserted BBC target.
            try:
                engine.insert_targets(copy.deepcopy(cfg),edited_hosts,tokenizer)
            except ValueError as e:
                assert "already contains target" in str(e), str(e)
                docs,infos,recs=first_docs,[],[]
                result.update(status="PASS",checks=["second insertion of existing BBC span rejected"],error={"expected":True,"type":"ValueError","message":str(e)})
            else:
                raise AssertionError("Engine accepted a second insertion of the same BBC target")
        else:
            actual_hosts = [] if kind == "empty_hosts" else hosts
            if expected.get("error") and kind == "engine":
                try:
                    engine.insert_targets(copy.deepcopy(cfg), copy.deepcopy(actual_hosts), tokenizer)
                except Exception as exc:
                    if type(exc).__name__ != expected["error"]:
                        raise
                    result.update(
                        status="PASS",
                        checks=["Engine raised the expected exception"],
                        error={"expected": True, "type": type(exc).__name__, "message": str(exc)},
                    )
                else:
                    raise AssertionError("Expected engine error but insertion succeeded")
            else:
                docs,infos,recs = engine.insert_targets(copy.deepcopy(cfg),copy.deepcopy(actual_hosts),tokenizer)
                checks=run_checks(hosts,docs,infos,recs,tokenizer,{**expected,"config":cfg})
                result.update(status="PASS",checks=checks)
    except Exception as exc:
        result.update(error={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
    # Complete original and modified text is retained inside each individual report.
    # If a test rejects invalid input, modified_dataset is null by design.
    write_json(out/(case["name"]+".json"),{
        "dataset_provenance":provenance,"test":case["description"],"configuration":cfg,
        "original_hosts":hosts,"modified_dataset":docs,"targets":infos,"insertions":recs,"result":result
    })
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",default="test cases")
    ap.add_argument("--only",default=None,help="Optional substring for selecting test cases")
    ap.add_argument("--print-docs",action="store_true",help="Print full original and modified text for changed documents to terminal (also always in JSON)")
    args=ap.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer
    try:
        hosts,bbc=extract_realtime_sources()
        tokenizers = {name: AutoTokenizer.from_pretrained(name, use_fast=True) for name in ("gpt2", "bert-base-uncased")}
        if not all(tk.is_fast for tk in tokenizers.values()):
            raise RuntimeError("Both models must provide fast tokenizers")
    except Exception as exc:
        write_json(out/"load_error.json",{"status":"ERROR","message":str(exc),"traceback":traceback.format_exc()})
        print("Real dataset/tokenizer load failed; see load_error.json",file=sys.stderr)
        return 2
    source_info={"hosts":{"dataset":PILE,"split":"train","source_ids":list(range(10)),"count":10},
                 "targets":{"dataset":BBC,"split":"train","sampled_rows":[{"row":r["row"],"label":r["label"],"sha256":engine.sha(r["text"])} for r in bbc]},"tokenizers":["gpt2", "bert-base-uncased"],"seed":42}
    write_json(out/"real_input_documents.json",{"provenance":source_info,"pile_hosts":hosts,"bbc_source_rows":bbc})
    cases_with_tokenizers=[]
    for tokenizer_name,tokenizer in tokenizers.items():
        cases,_=build_cases(hosts,bbc,tokenizer)
        for case in cases:
            case=copy.deepcopy(case)
            case["name"]=tokenizer_name.replace("-", "_")+"__"+case["name"]
            case["tokenizer"]=tokenizer_name
            cases_with_tokenizers.append(case)
    if args.only:
        cases_with_tokenizers=[c for c in cases_with_tokenizers if args.only in c["name"]]
    results=[]
    for j,c in enumerate(cases_with_tokenizers,1):
        tokenizer=tokenizers[c["tokenizer"]]
        r=execute(c,hosts,tokenizer,out,{**source_info,"case_tokenizer":c["tokenizer"]})
        results.append(r)
        print(f"[{j:03d}/{len(cases_with_tokenizers):03d}] {r['status']}: {c['name']}",flush=True)
        if args.print_docs and r["status"] == "PASS":
            data = json.loads((out / (c["name"] + ".json")).read_text(encoding="utf-8"))
            if isinstance(data.get("modified_dataset"), list):
                for old, new in zip(data["original_hosts"], data["modified_dataset"]):
                    if old["text"] != new["text"]:
                        print(f"\n--- ORIGINAL PILE DOC {old['source_id']} ---\n{old['text']}\n--- MODIFIED PILE DOC {old['source_id']} ---\n{new['text']}\n", flush=True)
    summary={"generated_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),"datasets":source_info,"total":len(results),"passed":sum(r["status"]=="PASS" for r in results),"failed":sum(r["status"]=="FAIL" for r in results),"results":results}
    write_json(out/"summary.json",summary)
    lines=["# Real-data controlled insertion test results","",f"- Pile hosts: **10 real documents** (source IDs 0-9)","- Target source: **SetFit/bbc-news** (real Hugging Face articles)","- Tokenizers: **GPT-2 fast**, **BERT base uncased fast**",f"- Result: **{summary['passed']}/{summary['total']} passed; {summary['failed']} failed**","","| Test case | Status | Description |","|---|---|---|"]
    lines.extend(f"| `{r['case']}` | {r['status']} | {r['description']} |" for r in results)
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"\nRESULT: {summary['passed']} PASS / {summary['failed']} FAIL. Files: {out.resolve()}")
    return 1 if summary["failed"] else 0


if __name__=="__main__":
    raise SystemExit(main())
