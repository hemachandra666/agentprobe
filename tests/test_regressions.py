"""Behavioral regressions from the repository reviews. No model required."""
import io
import json
import time
from types import SimpleNamespace
import pytest
from agentprobe import engine, scorer, student_agent, compare, agent, tools
from agentprobe.trajectory import Trajectory
from agentprobe.tasks_io import load_train, load_test
from agentprobe.prepare_training import to_examples

TASK = SimpleNamespace(task_id="x", family="test", question="Add 3 and 4 then multiply by 2",
                       answer=14.0, min_steps=2, reference_tools=["add", "multiply"])

class Scripted(engine.Provider):
    def __init__(self, actions): self.actions = iter(actions)
    def act(self, question, history): return next(self.actions)

def call(name="add", args=None):
    return engine.Action(name, args if args is not None else {"a":3,"b":4}, None, "call")

def final(value="14"):
    return engine.Action(None,None,value,value)

def correct_run(answer="14"):
    return engine.run("x", TASK.question, Scripted([call(), call("multiply", {"a":7,"b":2}), final(answer)]))

def test_wrong_final_does_not_inherit_correct_tool_result():
    t=correct_run("999")
    assert scorer.tool_result_correct(t,TASK)
    assert not scorer.answer_correct(t,TASK)
    assert not scorer.task_success(t,TASK)

def test_exhausted_run_is_not_completed():
    t=engine.run("x","q",Scripted([call(),call("multiply",{"a":7,"b":2})]),max_steps=2)
    assert t.termination == "max_steps"
    assert not scorer.task_success(t,TASK)

def test_last_call_budget_still_allows_final_answer():
    t=engine.run("x","q",Scripted([call(),call("multiply",{"a":7,"b":2}),final()]),max_calls=2)
    assert scorer.task_success(t,TASK)

def test_tool_budget_never_executes_extra_call(monkeypatch):
    executed=[]
    monkeypatch.setitem(tools.REGISTRY,"add",lambda **kw:executed.append(kw) or 7)
    t=engine.run("x","q",Scripted([call(),call(),call()]),max_calls=2)
    assert len(executed)==2
    assert t.termination=="max_calls"

def test_nested_bad_arguments_are_scorable():
    t=engine.run("x","q",Scripted([call(args={"a":[1],"b":2}),final()]))
    assert not scorer.score_one(t,TASK)["task_success"]
    assert t.steps[0].status=="error"

def test_deadline_stops_waiting_and_no_tool_executes(monkeypatch):
    class Slow(engine.Provider):
        def act(self,q,h): time.sleep(.2); return call()
    executed=[]
    monkeypatch.setitem(tools.REGISTRY,"add",lambda **kw:executed.append(kw) or 7)
    start=time.monotonic()
    t=engine.run("x","q",Slow(),max_seconds=.02)
    assert t.termination=="timeout"
    assert time.monotonic()-start < .15
    assert not executed and not t.steps

def test_repeated_calls_are_preserved(monkeypatch):
    class Student(student_agent._TextStudent):
        def _model(self): return None,None
    monkeypatch.setattr(student_agent,"_generate",lambda *a:"add(a=3,b=4)")
    p=Student()
    assert p.act("q",[]).tool==p.act("q",[]).tool=="add"
    t=engine.run("x","q",p,max_calls=2)
    assert len(t.steps)==2 and scorer.has_loop(t)

@pytest.mark.parametrize("text",["add(a=3,b=4)\nmultiply(a=7,b=2)","add(a=1,a=2)","add(x=1,b=2)","nonsense",""])
def test_invalid_text_is_not_final_answer(text):
    a=student_agent.parse_action(text)
    assert a.error and a.final_answer is None

def test_unknown_tool_is_recorded():
    t=engine.run("x","q",Scripted([student_agent.parse_action("divide(a=7,b=2)"),final()]))
    assert t.steps[0].tool=="divide" and t.steps[0].status=="error"

def test_scientific_notation_is_supported():
    a=student_agent.parse_action("multiply(a=-1e3, b=.5)")
    assert a.args=={"a":-1000.0,"b":.5}

def test_multiple_teacher_calls_are_not_silently_dropped(monkeypatch):
    monkeypatch.setattr(agent.model_client,"chat",lambda *a,**kw:{"message":{"tool_calls":[{},{}]}})
    assert agent.OllamaProvider().act("q",[]).error

def test_observations_are_visible_to_next_action():
    class Observer(engine.Provider):
        def act(self,q,h):
            if not h:return call()
            assert h[-1]==("observation","7.0")
            return final("7")
    assert engine.run("x","q",Observer()).final_answer=="7"

def test_run_serialization_keeps_failure_metadata(tmp_path):
    class Broken(engine.Provider):
        def act(self,q,h): raise RuntimeError("offline")
    t=engine.run("x","q",Broken());file=tmp_path/"nested"/"run.json";t.save(file)
    saved=json.loads(file.read_text())
    assert saved["termination"]=="provider_error" and saved["ended_at"] is not None
    assert saved["steps"][0]["kind"]=="provider"

def test_zero_call_comparison_still_records_attempt():
    output=io.StringIO()
    result=compare.eval_provider(lambda:Scripted([final()]),"mock",[TASK],1,"e",output)
    record=json.loads(output.getvalue())
    assert record["num_steps"]==0 and result["completion_rate"]==1
    assert result["task_success_rate"]==0
    assert result["error_rate_micro"] is None

def test_scoring_error_is_preserved(monkeypatch):
    output=io.StringIO()
    monkeypatch.setattr(scorer,"score_one",lambda *a:(_ for _ in ()).throw(ValueError("bad score")))
    with pytest.raises(RuntimeError):
        compare.eval_provider(lambda:Scripted([final()]),"mock",[TASK],1,"e",output)
    assert json.loads(output.getvalue())["score"]["scoring_error"]=="bad score"

def test_cycle_is_included_in_aggregate():
    output=io.StringIO()
    actions=[call(),call("multiply",{"a":7,"b":2}),call(),call("multiply",{"a":7,"b":2}),final()]
    result=compare.eval_provider(lambda:Scripted(actions),"mock",[TASK],1,"e",output)
    assert result["loop_any_rate"]==1

@pytest.mark.parametrize("args",[{"max_calls":0},{"max_steps":0},{"max_seconds":0}])
def test_invalid_budgets_fail_early(args):
    with pytest.raises(ValueError):engine.run("x","q",Scripted([]),**args)

def test_dataset_split_and_family_holdout():
    train,test=load_train(),load_test()
    assert not ({t.question for t in train}&{t.question for t in test})
    assert not ({t.family for t in train}&{"f_mul_add_mul","f_len5_b"})

def test_training_uses_same_feedback_format_as_inference():
    ex={"task_id":"x","question":"q","answer":14,"final_answer":"Answer: 14",
        "steps":[{"tool":"add","args":{"a":3,"b":4},"result":"7.0"},
                 {"tool":"multiply","args":{"a":7,"b":2},"result":"14.0"}]}
    rows=to_examples(ex)
    assert len(rows)==3
    assert rows[1]["messages"][-2]=={"role":"user","content":"result: 7.0"}
    assert rows[-1]["messages"][-1]["content"]=="Answer: 14.0"
    ex["final_answer"]="Answer: 999"
    assert to_examples(ex)==[]

@pytest.mark.parametrize("text",["multiply(a=7,b=2)","I tried 14 and 999", "nan", "Answer: inf"])
def test_nonanswers_are_not_numbers(text):
    assert scorer.final_numeric(text) is None

def test_evaluation_main_writes_versioned_results(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys,"argv",["run_eval","--models","teacher","--max-tasks","1","--output-dir",str(tmp_path)])
    monkeypatch.setattr(compare,"load_test",lambda directory:[TASK])
    monkeypatch.setattr(compare.agent,"OllamaProvider",lambda model:Scripted([call(),call("multiply",{"a":7,"b":2}),final()]))
    # Production Task is a dataclass; this fixture supplies equivalent serialization.
    monkeypatch.setattr(compare,"asdict",lambda task:vars(task))
    compare.main()
    result=json.loads(next(tmp_path.glob('*/comparison.json')).read_text())
    assert result["status"]=="complete" and result["scorer_version"]=="2.0"
    assert result["models"][0]["task_success_rate"]==1

def test_malformed_teacher_call_preserves_raw_response(monkeypatch):
    monkeypatch.setattr(agent.model_client,"chat",lambda *a,**kw:{"message":{"tool_calls":[{}]}})
    action=agent.OllamaProvider().act("q",[])
    assert action.error and 'tool_calls' in action.raw
