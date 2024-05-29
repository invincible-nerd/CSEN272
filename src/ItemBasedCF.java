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

        double[] trainSum=new double[201];
        int[] trainCount=new int[201];

        double[] testSum=new double[101];
        int[] testCount=new int[101];

        double[] trainAvg=new double[201]; //average rating of train user u2
        double[] testAvg=new double[101]; //average rating of test user u1

        for(int[] arr : train_list){
            int u=arr[0];
            int i=arr[1];
            train[u][i]=arr[2];
            trainCount[u]++;
            trainSum[u]+=arr[2];
        }

        for(int[] arr : test_list){
            if(arr[2]==0){
                continue;
            }
            int u=arr[0]-d;
            int i=arr[1];
            testSum[u]+=arr[2];
            testCount[u]++;
        }

        for(int u2=1; u2<=200; u2++){
            trainAvg[u2]=(double)trainSum[u2]/trainCount[u2];
        }
        for(int u1=1; u1<=100; u1++){
            testAvg[u1]=(double)testSum[u1]/testCount[u1];
        }


        for(int i=1; i<=1000; i++){
            for(int j=i+1; j<=1000; j++){
                double c=0;
                double a=0;
                double b=0;
                for(int u=1; u<=200; u++){
                    if(train[u][i]!=0 && train[u][j]!=0){
                        a+=(train[u][i]-trainAvg[u])*(train[u][i]-trainAvg[u]);
                        b+=(train[u][j]-trainAvg[u])*(train[u][j]-trainAvg[u]);
                        c+=(train[u][i]-trainAvg[u])*(train[u][j]-trainAvg[u]);
                    }
                }
                if(a!=0){
                    cos[i][j]=cos[j][i]=c/(Math.sqrt(a)*Math.sqrt(b));
                }
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
                    k++;
                }
                double pd=testAvg[u1]+sum/totalWeights;
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
