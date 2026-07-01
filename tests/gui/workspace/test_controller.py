from rheojax.gui.workspace.controller import Step, WorkflowController

def _ctl(ready):
    steps = [Step(id=str(i), title=str(i),
                  is_ready=(lambda i=i: ready[i]), validate=(lambda: True))
             for i in range(4)]
    return WorkflowController(steps)

def test_advance_gated_by_ready():
    # is_ready() means "this step's output is complete"; can_advance() checks CURRENT step
    ready = [False, False, False, False]
    c = _ctl(ready)
    assert c.current == 0 and c.can_advance() is False   # step 0 not done yet
    ready[0] = True
    assert c.can_advance() is True                        # step 0 done → can advance
    c.advance(); assert c.current == 1 and 1 in c.reached

def test_goto_only_reached():
    ready = [True, True, True, True]
    c = _ctl(ready)
    c.advance(); c.advance()                 # reached 0,1,2
    assert c.goto(1) is True and c.current == 1
    assert c.goto(3) is False                # 3 not reached yet

def test_edit_relocks_downstream():
    ready = [True, True, True, True]
    c = _ctl(ready)
    c.advance(); c.advance(); c.advance()     # reached {0,1,2,3}
    c.on_edit(1)                              # editing step 1
    assert c.reached == {0, 1}               # 2,3 re-locked
