import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

public class Util {


    Util(){
    }

    public List<int[]> read(String fileName){
        List<int[]> list=new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\s+"); // Split by whitespace
                int num1 = Integer.parseInt(parts[0]);
                int num2 = Integer.parseInt(parts[1]);
                int num3 = Integer.parseInt(parts[2]);
                list.add(new int[]{num1,num2,num3});
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return list;
    }

    public void write(List<int[]> res, String fileName){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            for (int[] array : res) {
                writer.write(array[0] + " " + array[1] + " " + array[2]);
                writer.newLine();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

//    public void writeS(List<String[]> res, String fileName){
//        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
//            for (String[] array : res) {
//                writer.write(String.join(" ", array));
//                writer.newLine();
//            }
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }

}
