function q = plotTargetRandom(objNum)
axis([-10 10 -10 10])
hold on

th = 0 : pi/60 : 2 *pi;

maxRad = 3;
minRad = 3;

maxLoci = 5;
minLoci = -5;

pos = zeros(objNum, 3);
for i = 1 : objNum
    pos(i,3) = (maxRad - minRad) * rand(1) + minRad;
    pos(i,1:2) = [(maxLoci - minLoci) * rand(1) + minLoci, (maxLoci - minLoci) * rand(1) + minLoci ];
    xunit = pos(i,3) * cos(th) + pos(i,1);
    yunit = pos(i,3) * sin(th) + pos(i,2);
    plot(xunit, yunit, 'r-'); 
end
q = reshape(pos', 1, size(pos,1) * 3);


maxTx = 0; 
minTx = -9;
maxTy = 0;
minTy = -9;
while 1
    x = (maxTx - minTx) * rand(1) + minTx;
    y = (maxTy - minTy) * rand(1) + minTy;
    isoutside = true;
    for i = 1 : objNum
        dist = (x - pos(i,1))^2 + (y - pos(i,2))^2;
        if dist < pos(i,3)^2
            isoutside = false;
            break;
        end
    end
    
    if isoutside
        break;
    end
end

plot(x, y, 'ro');
q = [q, x, y];

maxTx = 9; 
minTx = 0;
maxTy = 9;
minTy = 0;
while 1
    x = (maxTx - minTx) * rand(1) + minTx;
    y = (maxTy - minTy) * rand(1) + minTy;
    isoutside = true;
    for i = 1 : objNum
        dist = (x - pos(i,1))^2 + (y - pos(i,2))^2;
        if dist < pos(i,3)^2
            isoutside = false;
            break;
        end
    end
    
    if isoutside
        break;
    end
end

plot(x, y, 'bo');

q = [q, x, y];


end

