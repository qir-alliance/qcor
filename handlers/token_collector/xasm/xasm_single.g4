
grammar xasm_single;

line
   : statement 
   | comment
   ;

/* A program statement */
statement
   : qinst 
   | cinst
   ;

/* A program comment */
comment
   : COMMENT
   ;

qinst
   : inst_name=id '(' explist ')' ';'
   ;

cinst
   : 'const'? type_name=cpp_type var_name=exp ('=' var_value=exp)? ';'
   | exp '++' ';'
   | exp '--' ';'
   | 'for' '(' cpp_type exp '=' exp ';' (exp compare exp)? ';' ((exp ('++' | '--')) | (('++' | '--') exp))?  ')' '{'?
   | '}'
   | exp '(' explist ')' ';'
   | 'if' '(' explist ')' '{'?
   | 'else' '(' explist ')' '{'?
   | 'const'? type_name=cpp_type var_name=exp '=' '(' exp '==' exp ')' '?' exp ':' exp ';'
   | 'break' ';'
   | 'return' ';'
   | exp '=' exp ';'
   ;

cpp_type 
   : 'auto' ('&'|'*')?
   | exp
   ;

compare 
   : '>' | '<' | '>=' | '<=' ;

explist
   : exp ( ',' exp )*
   ;

exp
   : id
   | exp '+' exp
   | exp '-' exp
   | exp '*' exp
   | exp '/' exp
   | exp '::' exp
   | exp '<<' exp
   | exp '<' exp '>'
   | exp '::' exp '(' explist ')'
   | exp '.' exp '(' ')'
   | exp '.' exp '(' explist ')'
   | '-'exp
   | exp '^' exp
   | '(' exp ')'
   | unaryop '(' exp ')'
   | exp '[' exp ']'
   | string
   | real
   | INT
   | 'pi'
   | exp '&&' exp
   | exp '||' exp
   | '!' exp
   ;

unaryop
   : 'sin'
   | 'cos'
   | 'tan'
   | 'exp'
   | 'ln'
   | 'sqrt'
   ;

id
   : ID
   ;

real
   : REAL
   ;

string
   : STRING
   ;

COMMENT
   : '//' ~ [\r\n]* EOL
   ;

ID
   : [a-z][A-Za-z0-9_]*
   | [A-Z][A-Za-z0-9_]*
   | [A-Z][A-Za-z]*
   ;

REAL
   : INT? ( '.' (INT) )
   ;

INT
   : ('0'..'9')+
   ;

STRING
   : '"' ~ ["]* '"'
   ;

WS
   : [ \t\r\n] -> skip
   ;

EOL
: '\r'? '\n'
;
