import java.util.*;

public class ItemBasedCF {
    private int K;

    ItemBasedCF(int K){
        this.K=K;
    }

    private class W{
        int r;
        double weight;
        double abs;
        W(int r, double weight){
            this.r=r;
            this.weight=weight;
            this.abs=Math.abs(weight);
        }
    }

    List<int[]> adjustedCosineSimilarity(List<int[]> train_list, List<int[]> test_list, int d){
        int[][] train=new int[201][1001];

        double[][] cos=new double[1001][1001]; //cosine similarity of movie i, j

        int[] movieCount=new int[1001];
        int[] movieSum=new int[1001];
        double[] movieAvg=new double[1001];

        for(int[] arr : train_list){
            int u=arr[0];
            int i=arr[1];
            train[u][i]=arr[2];
            movieCount[i]++;
            movieSum[i]+=arr[2];
        }

        for(int i=1; i<=1000; i++){
            if(movieCount[i]>0){
                movieAvg[i]=(double)movieSum[i]/movieCount[i];
            }
        }

        for(int i=1; i<=1000; i++){
            if(movieCount[i]==0){
                continue;
            }
            for(int j=i+1; j<=1000; j++){
                if(movieCount[j]==0){
                    continue;
                }
                double c=0;
                double a=0;
                double b=0;
                for(int u=1; u<=200; u++){
                    if(train[u][i]!=0 && train[u][j]!=0){
                        a+=(train[u][i]-movieAvg[i])*(train[u][i]-movieAvg[i]);
                        b+=(train[u][j]-movieAvg[j])*(train[u][j]-movieAvg[j]);
                        c+=(train[u][i]-movieAvg[i])*(train[u][j]-movieAvg[j]);
                    }
                }
                cos[i][j]=cos[j][i]=c/(Math.sqrt(a)*Math.sqrt(b));
            }
        }

        Map<Integer,Integer>[] sets=new Map[101];
        for(int i=1; i<=100; i++){
            sets[i]=new HashMap<>();
        }
        for(int[] arr : test_list){
            if(arr[2]!=0){
                int u1=arr[0]-d;
                int i=arr[1];
                sets[u1].put(i,arr[2]);
            }
        }

        List<int[]> res=new ArrayList<>();
        for(int[] arr : test_list){
            if(arr[2]==0){
                int u1=arr[0]-d;
                int i=arr[1];
                double sum=0;
                double totalWeights=0;
                PriorityQueue<W> pq=new PriorityQueue<>((x, y)->(Double.compare(y.abs,x.abs)));
                Map<Integer,Integer> map=sets[u1];
                for(int j : map.keySet()){
                    if(cos[i][j]!=0){
                        pq.offer(new W(map.get(j),cos[i][j]));
                    }
                }
                int k=0;
                while(k<K && !pq.isEmpty()){
                    W w=pq.poll();
                    int r=w.r;
                    double weight=w.weight;
                    double abs=w.abs;
                    sum+=r*weight;
                    totalWeights+=abs;
                }
                double pd=sum/totalWeights;
                int p=5;

                if(pd<1.5){
                    p=1;
                }else if(pd<2.5){
                    p=2;
                }else if(pd<3.5){
                    p=3;
                }else if(pd<4.5){
                    p=4;
                }
                res.add(new int[]{u1,i,p});
            }
        }
        return res;
    }

}
