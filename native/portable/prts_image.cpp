#include "prts_image.h"
#include <algorithm>
#include <cmath>
#include <vector>
namespace {
struct Sample {int a,b;float f;};
Sample position(int index,int source,int target) {
    float value=float((index+.5)*double(source)/target-.5);int a=int(std::floor(value));float f=value-a;
    if(a<0){a=0;f=0;}
    if(a>=source-1){a=source-1;f=0;}
    return {a,std::min(a+1,source-1),f};
}
int rounded(double x){return int(std::nearbyint(x));}
}

int32_t prts_letterbox_rgb(const uint8_t *rgb,int32_t w,int32_t h,int32_t tw,int32_t th,
    const float *mean,const float *std,float *out,int32_t *crop) {
    if(!rgb||!out||!crop||w<1||h<1||tw<1||th<1||tw>4096||th>4096)return -1;
    double scale=std::min(double(tw)/w,double(th)/h);int rw=rounded(w*scale),rh=rounded(h*scale);
    if(rw<1||rh<1)return -1;
    int ox=(tw-rw)/2,oy=(th-rh)/2;crop[0]=ox;crop[1]=oy;crop[2]=rw;crop[3]=rh;
    for(int c=0;c<3;++c) {
        float m=mean?mean[c]:0.f,s=std?std[c]:1.f;if(s==0)return -1;
        std::fill(out+c*tw*th,out+(c+1)*tw*th,(114.f/255.f-m)/s);
    }
    std::vector<Sample> xs(rw),ys(rh);
    for(int x=0;x<rw;++x)xs[x]=position(x,w,rw);
    for(int y=0;y<rh;++y)ys[y]=position(y,h,rh);
    for(int y=0;y<rh;++y)for(int x=0;x<rw;++x)for(int c=0;c<3;++c) {
        auto px=xs[x],py=ys[y];
        float top=rgb[(py.a*w+px.a)*3+c]*(1-px.f)+rgb[(py.a*w+px.b)*3+c]*px.f;
        float bottom=rgb[(py.b*w+px.a)*3+c]*(1-px.f)+rgb[(py.b*w+px.b)*3+c]*px.f;
        float value=float(std::floor(top*(1-py.f)+bottom*py.f+.5f));
        out[c*tw*th+(y+oy)*tw+x+ox]=(value/255.f-(mean?mean[c]:0.f))/(std?std[c]:1.f);
    }
    return 0;
}

int32_t prts_decode_semantic(const float *scores,int32_t nc,int32_t sw,int32_t sh,const int32_t *crop,
    int32_t iw,int32_t ih,int32_t gw,int32_t gh,const int32_t *walk_ids,int32_t walk_count,
    uint8_t *ids,float *walk,float *confidence,uint8_t *walkable) {
    if(!scores||!crop||!ids||!walk||!confidence||!walkable||nc<1||nc>256||sw<1||sh<1||iw<1||ih<1||
       gw<1||gh<1||gw>4096||gh>4096||walk_count<0||(walk_count&&!walk_ids))return -1;
    int x0=rounded(double(crop[0])*sw/iw),y0=rounded(double(crop[1])*sh/ih);
    int x1=rounded(double(crop[0]+crop[2])*sw/iw),y1=rounded(double(crop[1]+crop[3])*sh/ih);
    if(x0<0||y0<0||x1>sw||y1>sh||x1<=x0||y1<=y0)return -1;
    std::vector<uint8_t> is_walk(nc,0);for(int j=0;j<walk_count;++j) {
        if(walk_ids[j]<0||walk_ids[j]>=nc)return -1;is_walk[walk_ids[j]]=1;
    }
    std::vector<Sample> xs(gw),ys(gh);
    for(int x=0;x<gw;++x)xs[x]=position(x,x1-x0,gw);
    for(int y=0;y<gh;++y)ys[y]=position(y,y1-y0,gh);
    for(int y=0;y<gh;++y)for(int x=0;x<gw;++x) {
        auto px=xs[x],py=ys[y];float sum=0.f,wprob=0.f,best=-1.f;int best_id=0;
        for(int c=0;c<nc;++c) {
            const float *plane=scores+c*sw*sh;
            float top=plane[(y0+py.a)*sw+x0+px.a]*(1-px.f)+plane[(y0+py.a)*sw+x0+px.b]*px.f;
            float bottom=plane[(y0+py.b)*sw+x0+px.a]*(1-px.f)+plane[(y0+py.b)*sw+x0+px.b]*px.f;
            float value=top*(1-py.f)+bottom*py.f;
            sum+=value;if(is_walk[c])wprob+=value;
            if(value>best){best=value;best_id=c;}
        }
        sum=std::max(sum,1e-8f);int i=y*gw+x;ids[i]=uint8_t(best_id);
        walk[i]=wprob/sum;confidence[i]=best/sum;walkable[i]=is_walk[best_id]&&walk[i]>=.50f;
    }
    return 0;
}
