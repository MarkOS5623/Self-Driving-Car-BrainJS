function lerp(A,B,t){ // returns a number between two numbers used mainly to manipulate baises and weights
    return A+(B-A)*t;
}

function getIntersection(A,B,C,D){ 
    const tTop=(D.x-C.x)*(A.y-C.y)-(D.y-C.y)*(A.x-C.x);
    const uTop=(C.y-A.y)*(A.x-B.x)-(C.x-A.x)*(A.y-B.y);
    const bottom=(D.y-C.y)*(B.x-A.x)-(D.x-C.x)*(B.y-A.y);
    
    if(bottom!=0){
        const t=tTop/bottom;
        const u=uTop/bottom;
        if(t>=0 && t<=1 && u>=0 && u<=1){
            return {
                x:lerp(A.x,B.x,t),
                y:lerp(A.y,B.y,t),
                offset:t
            }
        }
    }

    return null;
}

function polysIntersect(poly1, poly2){ // checks if two different polygons cross each other
    for(let i = 0; i < poly1.length; i++){
        for(let j = 0; j < poly2.length; j++){
            const touch = getIntersection(
                poly1[i],
                poly1[(i + 1) % poly1.length],
                poly2[j],
                poly2[(j + 1) % poly2.length]
            );
            if(touch){
                return true;
            }
        }
    }
    return false;
}

function getRGBA(value){
    const alpha = Math.abs(value);
    const R = value < 0 ? 0 : 255;
    const G = R;
    const B = value > 0 ? 0 : 255;
    return "rgba("+ R + "," + G + "," + B + "," + alpha + ")";
}
               
function saveLog() {
    var link = document.createElement('a');
    link.href = 'data:text/plain;charset=UTF-8,' + encodeURIComponent(file);
    link.innerHTML = 'download';
    link.download = 'Console_log.txt';     
    document.body.appendChild(link);
    link.click();
    link.remove();
}