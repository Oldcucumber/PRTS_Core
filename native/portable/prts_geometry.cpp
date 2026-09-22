#include "prts_geometry.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using Pixel=std::pair<int,int>;
int nearest(double value) { return static_cast<int>(std::nearbyint(value)); }
bool dimensions(int w,int h) { return w>=2 && h>=2 && w<=4096 && h<=4096; }

std::vector<uint8_t> inset(const uint8_t *mask,int w,int h) {
    std::vector<uint8_t> result(size_t(w)*h,0);
    for(int y=0;y<h;++y) {
        int radius=std::max(1,nearest(w*(.004+.046*std::max(0.,(double(y)/h-.45)/.50))));
        int blocked=0;
        for(int x=0;x<w;++x) {
            if(x==0) {
                for(int k=-radius;k<=radius;++k)
                    blocked+=(k<0 || k>=w || !mask[y*w+k]);
            } else {
                int old=x-radius-1,next=x+radius;
                blocked-=(old<0 || old>=w || !mask[y*w+old]);
                blocked+=(next<0 || next>=w || !mask[y*w+next]);
            }
            result[y*w+x]=(blocked==0);
        }
    }
    return result;
}

// 3x3 L2 chamfer, same weights as the reference OpenCV DIST_L2 / mask=3.
// Caller supplies a zero border when finite edge clearance is needed.
std::vector<float> chamfer(const std::vector<uint8_t>& mask,int w,int h) {
    const float a=.955f,b=1.3693f,inf=1e20f;
    std::vector<float> d(mask.size());
    for(size_t i=0;i<mask.size();++i)d[i]=mask[i]?inf:0.f;
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        int i=y*w+x;if(!mask[i])continue;
        float value=d[i];
        if(x)value=std::min(value,d[i-1]+a);
        if(y) {
            value=std::min(value,d[i-w]+a);
            if(x)value=std::min(value,d[i-w-1]+b);
            if(x+1<w)value=std::min(value,d[i-w+1]+b);
        }
        d[i]=value;
    }
    for(int y=h-1;y>=0;--y)for(int x=w-1;x>=0;--x) {
        int i=y*w+x;if(!mask[i])continue;
        float value=d[i];
        if(x+1<w)value=std::min(value,d[i+1]+a);
        if(y+1<h) {
            value=std::min(value,d[i+w]+a);
            if(x)value=std::min(value,d[i+w-1]+b);
            if(x+1<w)value=std::min(value,d[i+w+1]+b);
        }
        d[i]=value;
    }
    return d;
}

std::vector<float> clearance(const uint8_t *mask,int w,int h) {
    std::vector<uint8_t> padded(size_t(w+2)*(h+2),0);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)padded[(y+1)*(w+2)+x+1]=mask[y*w+x];
    auto dist=chamfer(padded,w+2,h+2);std::vector<float> result(size_t(w)*h);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)result[y*w+x]=dist[(y+1)*(w+2)+x+1];
    return result;
}

std::set<Pixel> raster(const int32_t *xy,int count) {
    std::set<Pixel> pixels;
    if(count==1)pixels.emplace(xy[0],xy[1]);
    for(int i=1;i<count;++i) {
        int ax=xy[2*i-2],ay=xy[2*i-1],bx=xy[2*i],by=xy[2*i+1];
        int n=std::max({std::abs(bx-ax),std::abs(by-ay),1});
        for(int t=0;t<=n;++t)pixels.emplace(nearest(ax+(bx-ax)*double(t)/n),nearest(ay+(by-ay)*double(t)/n));
    }
    return pixels;
}
}

int32_t prts_corridor_search(const uint8_t *mask,int32_t w,int32_t h,int32_t anchor,
    int32_t prefer,const int32_t *prior,int32_t prior_count,int32_t *out,int32_t capacity,double *coverage) {
    if(!mask || !out || !coverage || !dimensions(w,h) || capacity<0 || prior_count<0 ||
       (prior_count && !prior) || prefer<-1 || prefer>1)return -1;
    *coverage=0.;const int sy=int(h*.94),ey=int(h*.42);auto safe=inset(mask,w,h);
    if(anchor<0) {
        double closest=std::numeric_limits<double>::infinity();
        for(int x=0;x<w;++x)if(safe[sy*w+x] && std::abs(x-w/2.)<w*.08 && std::abs(x-w/2.)<closest) {
            closest=std::abs(x-w/2.);anchor=x;
        }
    }
    if(anchor<0 || anchor>=w || !safe[sy*w+anchor])return 0;
    for(int y=0;y<h;++y)if(y<ey || y>sy)std::fill(safe.begin()+y*w,safe.begin()+(y+1)*w,0);
    std::vector<uint8_t> connected(size_t(w)*h,0);std::queue<Pixel> flood;
    flood.emplace(anchor,sy);connected[sy*w+anchor]=1;int gy=sy;
    const int dx4[]={-1,1,0,0},dy4[]={0,0,-1,1};
    while(!flood.empty()) {
        auto [x,y]=flood.front();flood.pop();gy=std::min(gy,y);
        for(int j=0;j<4;++j) {
            int nx=x+dx4[j],ny=y+dy4[j];
            if(nx>=0 && nx<w && ny>=ey && ny<=sy && safe[ny*w+nx] && !connected[ny*w+nx]) {
                connected[ny*w+nx]=1;flood.emplace(nx,ny);
            }
        }
    }
    if(gy==sy)return 0;
    auto distance=clearance(mask,w,h);double target=prefer?w*(.5+.26*prefer):anchor;
    double best=-std::numeric_limits<double>::infinity();int gx=-1;
    for(int x=0;x<w;++x)if(connected[gy*w+x]) {
        double score=double(distance[gy*w+x])/w-.20*std::abs(x-target)/w;
        if(score>best){best=score;gx=x;}
    }
    std::vector<float> prior_distance;
    if(prior_count) {
        std::vector<uint8_t> line(size_t(w)*h,1);
        // Reference previous paths are dense adjacent pixels; the full-polyline
        // raster also accepts sparse caller paths without opening free space.
        for(auto [x,y]:raster(prior,prior_count))if(x>=0&&x<w&&y>=0&&y<h)line[y*w+x]=0;
        prior_distance=chamfer(line,w,h);
    }
    using Node=std::tuple<double,double,int,int>;
    std::priority_queue<Node,std::vector<Node>,std::greater<Node>> queue;
    std::vector<double> cost(size_t(w)*h,std::numeric_limits<double>::infinity());
    std::vector<int> parent(size_t(w)*h,-1);
    cost[sy*w+anchor]=0.;queue.emplace(std::hypot(gx-anchor,sy-gy),0.,anchor,sy);
    const int dx[]={-1,0,1,-1,1,-1,0,1},dy[]={-1,-1,-1,0,0,1,1,1};bool reached=false;
    while(!queue.empty()) {
        auto [unused,g,x,y]=queue.top();queue.pop();
        if(g>cost[y*w+x]+1e-8)continue;
        if(x==gx&&y==gy){reached=true;break;}
        for(int j=0;j<8;++j) {
            int nx=x+dx[j],ny=y+dy[j];
            if(nx<0 || nx>=w || ny<ey || ny>sy || !connected[ny*w+nx])continue;
            if(dx[j]&&dy[j]&&(!safe[y*w+nx]||!safe[ny*w+x]))continue;
            double extra=.8/(1+double(distance[ny*w+nx]));
            if(!prior_distance.empty())extra+=.15*std::min(.3,double(prior_distance[ny*w+nx])/w);
            double step=dx[j]&&dy[j]?1.4142135623730951:1.;double ng=g+step*(1+extra);
            if(ng+1e-8<cost[ny*w+nx]) {
                cost[ny*w+nx]=ng;parent[ny*w+nx]=y*w+x;
                queue.emplace(ng+std::hypot(gx-nx,gy-ny),ng,nx,ny);
            }
        }
    }
    if(!reached)return 0;
    std::vector<Pixel> path;int p=gy*w+gx;
    while(true) {
        path.emplace_back(p%w,p/w);if(p==sy*w+anchor)break;p=parent[p];
    }
    if(path.size()>size_t(capacity))return -2;
    std::reverse(path.begin(),path.end());
    for(size_t i=0;i<path.size();++i){out[2*i]=path[i].first;out[2*i+1]=path[i].second;}
    *coverage=double(sy-gy)/std::max(1,sy-ey);return int32_t(path.size());
}

int32_t prts_path_validate(const uint8_t *mask,int32_t w,int32_t h,const int32_t *xy,int32_t count,int32_t *checked) {
    if(!mask || !checked || !dimensions(w,h) || count<0 || (count&&!xy))return -1;
    auto pixels=raster(xy,count);*checked=int32_t(pixels.size());int outside=0;
    for(auto [x,y]:pixels)outside+=(x<0 || x>=w || y<0 || y>=h || !mask[y*w+x]);
    return outside;
}

int32_t prts_path_heading(const int32_t *xy,int32_t count,int32_t w,int32_t h,double *angles) {
    if(!xy || !angles || count<1 || !dimensions(w,h))return -1;
    int min_y=h;for(int i=0;i<count;++i)min_y=std::min(min_y,xy[2*i+1]);
    auto at=[&](double row) {
        double nearest_y=std::numeric_limits<double>::infinity();
        for(int i=0;i<count;++i)nearest_y=std::min(nearest_y,std::abs(xy[2*i+1]-row*h));
        std::vector<int> xs;for(int i=0;i<count;++i)if(std::abs(xy[2*i+1]-row*h)<=nearest_y+1)xs.push_back(xy[2*i]);
        std::sort(xs.begin(),xs.end());size_t n=xs.size();return (double(xs[(n-1)/2])+xs[n/2])/(2*w);
    };
    double near=.87,far=std::max(.57,double(min_y)/h),mid=(near+far)/2,k=180/3.14159265358979323846;
    angles[0]=std::atan2(at(far)-at(near),near-far)*k;
    angles[1]=std::atan2(at(mid)-at(near),near-mid)*k;
    angles[2]=std::atan2(at(far)-at(mid),mid-far)*k;return 0;
}
