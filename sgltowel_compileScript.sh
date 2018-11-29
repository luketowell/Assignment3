usage(){
        echo $0
}

if [[ $# -ne 0 ]]; then
        echo You have entered too many arguments, see usage below.
        usage
        exit -1
fi

EXE=sgltowel_code.cu
OUTPUTFILE=sgltowel-code.exe
COMPILECOMMAND = nvcc ${EXE} -o ${OUTPUTFILE}

echo running ${COMPILECOMMAND}
$COMPILECOMMAND
echo $OUTPUTFILE file compiled