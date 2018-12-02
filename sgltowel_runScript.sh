usage(){
        echo $0  numOMPThreads numGPUThreads arrayLength 
}

if [[ $# -ne 3 ]]; then
        usage
        exit -1
fi

EXE=sgltowel_code.exe
OMPTHREADS=$1
GPUTHREADS=$2
DATAPOINTS=30000000

export OMP_NUM_THREADS=$OMPTHREADS
echo executing file: ${EXE}
./${EXE} ${DATAPOINTS} ${GPUTHREADS} ${OMPTHREADS}            

echo Finished