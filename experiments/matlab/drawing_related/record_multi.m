function record(object, eventdata)
    global mouseDown
    global trajectories
    global trajNum

    trajNum = trajNum + 1;
    mouseDown = true;
    trajectories{trajNum} = [];
end

