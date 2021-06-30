package DiabetesDiagnosis;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ReadFile {

    private  List<String[]> allData= new ArrayList<>();
    private static File file = new File("src/DiabetesDiagnosis/DataFile/diabetes.csv");

    private  List<String[]> learningDataSet= new ArrayList<>();
    private  List<String[]> testingDataSet= new ArrayList<>();
    private  String[] label;




    public  void readCsv() throws IOException {
        FileReader fileReader = new FileReader(file);
        BufferedReader bufferedReader = new BufferedReader(fileReader);
        String line;
        String[] tokenizedLine;
        int count=0;
        while ((line = bufferedReader.readLine()) != null) {
            tokenizedLine = parse(line);
            count++;
            allData.add(tokenizedLine);
        }
    }




    public void showAllData(){
        for (String[] tab : allData) {
            for (int i = 0; i < tab.length; i++) {
                System.out.print(tab[i] + ",");
            }
            System.out.println();
        }
    }



    public void segregateData(){
        label=allData.get(0);
        int rowsNumber= allData.size()-1;
        int learningRows= ( rowsNumber /4 ) * 3  ;
        for( int i = 1 ; i <= learningRows ; i ++){
            learningDataSet.add(allData.get(i));
        }
        int testRows = rowsNumber- learningRows;
        for( int i = learningRows +1 ; i <= rowsNumber ; i ++){
            testingDataSet.add(allData.get(i));
        }
    }



    private static String[] parse(String line) { // use split or Scanner
        return line.split(",");
    }

    private static String getString(String [] arr){
        String str = String.join(",", arr);
        return str;
    }


    public List<String[]> getLearningDataSet() {
        return learningDataSet;
    }

    public List<String[]> getTestingDataSet() {
        return testingDataSet;
    }
}
