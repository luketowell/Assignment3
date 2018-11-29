usage(){
        echo $0
}

if [[ $# -ne 0 ]]; then
        echo You have entered too many arguments, see usage below.
        usage
        exit -1
fi

EXE=sgltowel_code.cu
OUTPUTFILE=sgltowel_code.exe

echo running ${COMPILECOMMAND}
echo load module cuda-8.0
module load cuda-8.0
nvcc -Xcompiler -fopenmp ./${EXE} -o ./${OUTPUTFILE}
echo $OUTPUTFILE file compiled