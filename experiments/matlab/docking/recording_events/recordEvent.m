function recordEvent(object, eventdata)
    global mouseDown trajectories trajNum queries cquery

    trajNum = trajNum + 1;
    mouseDown = true;
    trajectories{trajNum} = [];
    
    queries = [queries;cquery];
end

