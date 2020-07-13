function stop_multi_record(object, eventdata)
    global mouseDown queries genEnv;
    mouseDown = false;
    
    clf

    queries = [queries;plotRandOriGoalTask()];
end

