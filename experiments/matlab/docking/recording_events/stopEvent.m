function stopEvent(object, eventdata)
    global mouseDown cquery;
    mouseDown = false;
    
    clf
    cquery = plotDockingStations();
end

