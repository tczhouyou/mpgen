function mouseMove(object, eventdata)
    global trajectories;
    global mouseDown;
    global trajNum;
    
    C = get (gca, 'CurrentPoint');
    title(gca, ['(X,Y) = (', num2str(C(1,1)), ', ',num2str(C(1,2)), ')']);
    
    if mouseDown
        ctraj = trajectories{trajNum};
        ctraj = [ctraj; C(1,1), C(1,2)];
        plot(ctraj(:,1), ctraj(:,2), 'b-');        
        trajectories{trajNum} = ctraj;
    end
    
end