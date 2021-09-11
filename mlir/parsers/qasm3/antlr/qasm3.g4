/***** ANTLRv4  grammar for OpenQASM3.0. *****/

grammar qasm3;

/**** Parser grammar ****/

program
    : header (globalStatement | statement)*
    ;

header
    : version? include*
    ;

version
    : 'OPENQASM' ( Integer | RealNumber ) SEMICOLON
    ;

include
    : 'include' StringLiteral SEMICOLON
    ;

globalStatement
    : subroutineDefinition
    | kernelDeclaration
    | quantumGateDefinition
    | calibration
    | quantumDeclarationStatement  // qubits are declared globally
    | pragma
    ;


statement
    : expressionStatement
    | assignmentStatement
    | classicalDeclarationStatement
    | branchingStatement
    | loopStatement
    | controlDirectiveStatement
    | aliasStatement
    | quantumStatement
    | returnStatement
    | qcor_test_statement
    | compute_action_stmt
    ;

compute_action_stmt
    : 'compute' compute_block=programBlock 'action' action_block=programBlock
    ;

qcor_test_statement
    : 'QCOR_EXPECT_TRUE' LPAREN booleanExpression RPAREN SEMICOLON
    ;

quantumDeclarationStatement : quantumDeclaration SEMICOLON ;

classicalDeclarationStatement
    : ( classicalDeclaration | constantDeclaration ) SEMICOLON
    ;

classicalAssignment
    : indexIdentifier assignmentOperator ( expression | indexIdentifier )
    ;

assignmentStatement : ( classicalAssignment | quantumMeasurementAssignment ) SEMICOLON ;

returnSignature
    : ARROW classicalType
    ;

/*** Types and Casting ***/

designator
    : LBRACKET expression RBRACKET
    ;

doubleDesignator
    : LBRACKET expression COMMA expression RBRACKET
    ;

identifierList
    : ( Identifier COMMA )* Identifier
    ;

association
    : COLON Identifier
    ;

/** Quantum Types **/
quantumType
    : 'qubit'
    | 'qreg'
    ;

quantumDeclaration
    : 'qreg' Identifier designator? | 'qubit' designator? Identifier
    ;

quantumArgument
    : quantumType designator? association
    ;

quantumArgumentList
    : ( quantumArgument COMMA )* quantumArgument
    ;

/** Classical Types **/
bitType
    : 'bit'
    | 'creg'
    ;

singleDesignatorType
    : 'int'
    | 'uint'
    | 'float'
    | 'angle'
    ;

doubleDesignatorType
    : 'fixed'
    ;

noDesignatorType
    : 'bool'
    | timingType
    | 'int' | 'int64_t' | 'float' | 'double'
    ;

classicalType
    : singleDesignatorType designator
    | doubleDesignatorType doubleDesignator
    | noDesignatorType
    | bitType designator?
    ;

constantDeclaration
    : 'const' equalsAssignmentList
    ;

// if multiple variables declared at once, either none are assigned or all are assigned
// prevents ambiguity w/ qubit arguments in subroutine calls
singleDesignatorDeclaration
    : singleDesignatorType designator ( identifierList | equalsAssignmentList )
    ;

doubleDesignatorDeclaration
    : doubleDesignatorType doubleDesignator ( identifierList | equalsAssignmentList )
    ;

noDesignatorDeclaration
    : noDesignatorType ( identifierList | equalsAssignmentList )
    ;

bitDeclaration
    : bitType (indexIdentifierList | indexEqualsAssignmentList )
    ;

classicalDeclaration
    : singleDesignatorDeclaration
    | doubleDesignatorDeclaration
    | noDesignatorDeclaration
    | bitDeclaration
    ;

classicalTypeList
    : ( classicalType COMMA )* classicalType
    ;

classicalArgument
    : classicalType association
    ;

classicalArgumentList
    : ( classicalArgument COMMA )* classicalArgument
    ;

/** Aliasing **/
aliasStatement
    : 'let' Identifier EQUALS indexIdentifier SEMICOLON
    ;

/** Register Concatenation and Slicing **/

indexIdentifier
    : Identifier rangeDefinition
    | Identifier ( LBRACKET expressionList RBRACKET )?
    | indexIdentifier '||' indexIdentifier
    ;

indexIdentifierList
    : ( indexIdentifier COMMA )* indexIdentifier
    ;

indexEqualsAssignmentList
    : ( indexIdentifier equalsExpression COMMA)* indexIdentifier equalsExpression
    ;

rangeDefinition
    : LBRACKET expression? COLON expression? ( COLON expression )? RBRACKET
    ;

/*** Gates and Built-in Quantum Instructions ***/

quantumGateDefinition
    : 'gate' quantumGateSignature quantumBlock
    ;

quantumGateSignature
    : ( Identifier | 'CX' | 'U') ( LPAREN identifierList? RPAREN )? identifierList
    ;

quantumBlock
    : LBRACE ( compute_action_stmt | quantumStatement | quantumLoop )* RBRACE
    ;

// loops containing only quantum statements allowed in gates
quantumLoop
    : loopSignature quantumLoopBlock
    ;

quantumLoopBlock
    : quantumStatement
    | LBRACE quantumStatement* RBRACE
    ;

quantumStatement
    : quantumInstruction SEMICOLON
    | timingStatement
    ;

quantumInstruction
    : quantumGateCall
    | quantumPhase
    | quantumMeasurement
    | quantumBarrier
    ;

quantumPhase
    : 'gphase' LPAREN Identifier RPAREN
    ;

quantumMeasurement
    : 'measure' indexIdentifierList
    ;

quantumMeasurementAssignment
    : quantumMeasurement ( ARROW indexIdentifierList)?
    | indexIdentifierList EQUALS quantumMeasurement
    ;

quantumBarrier
    : 'barrier' indexIdentifierList
    ;

quantumGateModifier
    : ( 'inv' | 'pow' LPAREN expression RPAREN | 'ctrl' ) '@'
    ;

quantumGateCall
    : quantumGateName ( LPAREN expressionList? RPAREN )? indexIdentifierList
    ;

quantumGateName
    : 'CX'
    | 'U'
    | 'reset'
    | Identifier
    | quantumGateModifier quantumGateName
    ;

/*** Classical Instructions ***/

unaryOperator
    : '~' | '!'
    ;

relationalOperator
    : '>'
    | '<'
    | '>='
    | '<='
    | '=='
    | '!='
    ;

logicalOperator
    : '&&'
    | '||'
    ;

expressionStatement
    : expression SEMICOLON
    ;

expression
    // include terminator/unary as base cases to simplify parsing
    : expressionTerminator
    | unaryExpression
    // expression hierarchy
    | xOrExpression
    | expression '|' xOrExpression
    ;

/**  Expression hierarchy for non-terminators. Adapted from ANTLR4 C
  *  grammar: https://github.com/antlr/grammars-v4/blob/master/c/C.g4
  * Order (first to last evaluation):
    Terminator (including Parens),
    Unary Op,
    Multiplicative
    Additive
    Bit Shift
    Bit And
    Exlusive Or (xOr)
    Bit Or
**/
xOrExpression
    : bitAndExpression
    | xOrExpression '^' bitAndExpression
    ;

bitAndExpression
    : bitShiftExpression
    | bitAndExpression '&' bitShiftExpression
    ;

bitShiftExpression
    : additiveExpression
    | bitShiftExpression ( '<<' | '>>' ) additiveExpression
    ;

additiveExpression
    : multiplicativeExpression
    | additiveExpression binary_op=( '+' | MINUS ) multiplicativeExpression
    ;

multiplicativeExpression
    // base case either terminator or unary
    : expressionTerminator
    | unaryExpression
    | multiplicativeExpression binary_op=( '*' | '/' | '%' ) ( expressionTerminator | unaryExpression )
    ;

unaryExpression
    : unaryOperator expressionTerminator
    ;

expressionTerminator
    : Constant
    | Integer
    | RealNumber
    | Identifier
    | StringLiteral
    | builtInCall
    | kernelCall
    | subroutineCall
    | timingTerminator
    | MINUS expressionTerminator
    | LPAREN expression RPAREN
    | expressionTerminator LBRACKET expression RBRACKET
    | expressionTerminator incrementor
    ;
/** End expression hierarchy **/

incrementor
    : '++'
    | '--'
    ;

builtInCall
    : ( builtInMath | castOperator ) LPAREN expressionList RPAREN
    ;

builtInMath
    : 'sin' | 'cos' | 'tan' | 'arctan' | 'arccos' | 'arcsin' | 'exp' | 'ln' | 'sqrt' | 'rotl' | 'rotr' | 'popcount' | 'lengthof'
    ;

castOperator
    : classicalType
    ;

expressionList
    : ( expression COMMA )* expression
    ;

/** Boolean expression hierarchy **/
booleanExpression
    : membershipTest
    | comparsionExpression
    | booleanExpression logicalOperator comparsionExpression
    ;

comparsionExpression
    : expression  // if (expression)
    | expression relationalOperator expression
    ;
/** End boolean expression hierarchy **/

equalsExpression
    : EQUALS expression
    ;

assignmentOperator
    : EQUALS
    | '+=' | '-=' | '*=' | '/=' | '&=' | '|=' | '~=' | '^=' | '<<=' | '>>='
    ;

equalsAssignmentList
    : ( Identifier equalsExpression COMMA)* Identifier equalsExpression
    ;

membershipTest
    : Identifier 'in' setDeclaration
    ;

setDeclaration
    : LBRACE expressionList RBRACE
    | rangeDefinition
    | Identifier
    ;

programBlock
    : statement
    | LBRACE statement* RBRACE
    ;

branchingStatement
    : 'if' LPAREN booleanExpression RPAREN programBlock ( 'else' programBlock )?
    ;

loopSignature
    : 'for' membershipTest
    | 'while' LPAREN booleanExpression RPAREN
    ;

loopStatement: loopSignature programBlock ;

cLikeLoopStatement
    : 'for' LPAREN
            classicalType Identifier equalsExpression ';'
            booleanExpression ';'
            expression
            RPAREN
            LBRACE
              programBlock
            RBRACE
    | 'for' LPAREN
              classicalType Identifier ':' 'range' LPAREN expression RPAREN
            RPAREN
            LBRACE
              programBlock
            RBRACE
    ;

controlDirectiveStatement
    : controlDirective SEMICOLON
    ;

controlDirective
    : 'break'
    | 'continue'
    | 'end'
    ;

kernelDeclaration
    : 'kernel' Identifier ( LPAREN classicalTypeList? RPAREN )? returnSignature?
    classicalType? SEMICOLON
    ;

// if have kernel w/ out args, is ambiguous; may get matched as identifier
kernelCall
    : Identifier LPAREN expressionList? RPAREN
    ;

/*** Subroutines ***/

subroutineDefinition
    : 'def' Identifier ( LPAREN classicalArgumentList? RPAREN )? quantumArgumentList?
    returnSignature? subroutineBlock
    ;

returnStatement : 'return' statement;

subroutineBlock
    : LBRACE statement* returnStatement? RBRACE
    // Begin QCOR Extension
    // Allow a special 'extern' keyword in place of the subroutine definition
    // to denote a declaration of an externally-provided quantum subroutine.
    | EXTERN SEMICOLON
    // End QCOR Extension
    ;

// if have subroutine w/ out args, is ambiguous; may get matched as identifier
subroutineCall
    : Identifier ( LPAREN expressionList? RPAREN ) expressionList
    ;

/*** Directives ***/

pragma
    : '#pragma' LBRACE statement* RBRACE  // match any valid openqasm statement
    ;

/*** Circuit Timing ***/

timingType
    : 'length'
    | 'stretch' Integer?
    ;

timingBox
    : 'boxas' Identifier quantumBlock
    | 'boxto' TimingLiteral quantumBlock
    ;

timingTerminator
    : timingIdentifier | 'stretchinf'
    ;

timingIdentifier
    : TimingLiteral
    | 'lengthof' LPAREN ( Identifier | quantumBlock ) RPAREN
    ;

timingInstructionName
    : 'delay'
    | 'rotary'
    ;

timingInstruction
    : timingInstructionName ( LPAREN expressionList? RPAREN )? designator indexIdentifierList
    ;

timingStatement
    : timingInstruction SEMICOLON
    | timingBox
    ;

/*** Pulse Level Descriptions of Gates and Measurement ***/
// TODO: Update when pulse grammar is formalized

calibration
    : calibrationGrammarDeclaration
    | calibrationDefinition
    ;

calibrationGrammarDeclaration
    : 'defcalgrammar' calibrationGrammar SEMICOLON
    ;

calibrationDefinition
    : 'defcal' Identifier
    ( LPAREN calibrationArgumentList? RPAREN )? identifierList
    returnSignature? LBRACE .*? RBRACE  // for now, match anything inside body
    ;

calibrationGrammar
    : '"openpulse"' | StringLiteral  // currently: pulse grammar string can be anything
    ;

calibrationArgumentList
    : classicalArgumentList | expressionList
    ;

/**** Lexer grammar ****/

LBRACKET : '[' ;
RBRACKET : ']' ;

LBRACE : '{' ;
RBRACE : '}' ;

LPAREN : '(' ;
RPAREN : ')' ;

COLON: ':' ;
SEMICOLON : ';' ;

DOT : '.' ;
COMMA : ',' ;

EQUALS : '=' ;
ARROW : '->' ;

MINUS : '-' ;

// Begin QCOR Extension
EXTERN : 'extern' ;
// End QCOR Extension

Constant : ( 'pi' | 'π' | 'tau' | '𝜏' | 'euler' | 'ℇ' );

Whitespace : [ \t]+ -> skip ;
Newline : [\r\n]+ -> skip ;

fragment Digit : [0-9] ;
Integer : Digit+ ;

fragment ValidUnicode : [\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}] ; // valid unicode chars
fragment Letter : [A-Za-z] ;
fragment FirstIdCharacter : '_' | '$' | ValidUnicode | Letter ;
fragment GeneralIdCharacter : FirstIdCharacter | Integer;

Identifier : FirstIdCharacter GeneralIdCharacter* ;

fragment SciNotation : [eE] ;
fragment PlusMinus : [-+] ;
fragment Float : Digit+ DOT Digit* | DOT Digit*;
RealNumber : Float (SciNotation PlusMinus? Integer )? ;

fragment TimeUnit : 'dt' | 'ns' | 'us' | 'µs' | 'ms' | 's' ;
// represents explicit time value in SI or backend units
TimingLiteral : (Integer | RealNumber ) TimeUnit ;

// allow ``"str"`` and ``'str'``
StringLiteral
    : '"' ~["\r\t\n]+? '"'
    | '\'' ~['\r\t\n]+? '\''
    ;

// skip comments
LineComment : '//' ~[\r\n]* -> skip;
BlockComment : '/*' .*? '*/' -> skip;