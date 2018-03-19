/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#include "occa/tools/testing.hpp"

#include "parser.hpp"

void testPeek();
void testLoading();
void testErrors();

using namespace occa::lang;

//---[ Util Methods ]-------------------
std::string source;
parser_t parser;

void setSource(const std::string &s) {
  source = s;
  parser.setSource(s, false);
}

void parseSource(const std::string &s) {
  source = s;
  parser.parseSource(s);
}

template <class smntType>
smntType& getStatement(const int index = 0) {
  return parser.root[index]->to<smntType>();
}
//======================================

//---[ Macro Util Methods ]-------------
#define testStatementPeek(str_, type_)              \
  setSource(str_);                                  \
  OCCA_ASSERT_EQUAL_BINARY(type_,                   \
                           parser.peek());          \
  OCCA_ASSERT_TRUE(parser.success)

#define testStatementLoading(str_, type_)           \
  parseSource(str_);                                \
  OCCA_ASSERT_EQUAL(1,                              \
                    parser.root.size());            \
  OCCA_ASSERT_EQUAL_BINARY(type_,                   \
                           parser.root[0]->type())  \
  OCCA_ASSERT_TRUE(parser.success)
//======================================

int main(const int argc, const char **argv) {
  testPeek();
  testLoading();
  testErrors();

  return 0;
}

//---[ Peek ]---------------------------
void testPeek() {
  testStatementPeek("",
                    statementType::none);

  testStatementPeek("#pragma",
                    statementType::pragma);
  testStatementPeek("#pragma occa test",
                    statementType::pragma);

  testStatementPeek("1 + 2;",
                    statementType::expression);
  testStatementPeek("\"a\";",
                    statementType::expression);
  testStatementPeek("'a';",
                    statementType::expression);
  testStatementPeek("sizeof 3;",
                    statementType::expression);

  testStatementPeek("int a = 0;",
                    statementType::declaration);
  testStatementPeek("const int a = 0;",
                    statementType::declaration);
  testStatementPeek("long long a, b = 0, *c = 0;",
                    statementType::declaration);

  testStatementPeek("goto foo;",
                    statementType::goto_);

  testStatementPeek("foo:",
                    statementType::gotoLabel);

  testStatementPeek("{}",
                    statementType::block);

  testStatementPeek("namespace foo {}",
                    statementType::namespace_);

  testStatementPeek("if (true) {}",
                    statementType::if_);
  testStatementPeek("else if (true) {}",
                    statementType::elif_);
  testStatementPeek("else {}",
                    statementType::else_);

  testStatementPeek("for () {}",
                    statementType::for_);

  testStatementPeek("while () {}",
                    statementType::while_);
  testStatementPeek("do {} while ()",
                    statementType::while_);

  testStatementPeek("switch () {}",
                    statementType::switch_);
  testStatementPeek("case foo:",
                    statementType::case_);
  testStatementPeek("continue;",
                    statementType::continue_);
  testStatementPeek("break;",
                    statementType::break_);

  testStatementPeek("return 0;",
                    statementType::return_);

  testStatementPeek("@attr",
                    statementType::attribute);
  testStatementPeek("@attr()",
                    statementType::attribute);

  testStatementPeek("public:",
                    statementType::classAccess);
  testStatementPeek("protected:",
                    statementType::classAccess);
  testStatementPeek("private:",
                    statementType::classAccess);
}
//======================================

//---[ Loading ]------------------------
void testExpression();
void testDeclaration();
void testBlock();
void testNamespace();
void testTypeDecl();
void testIf();
void testFor();
void testWhile();
void testSwitch();
void testJumps();
void testClassAccess();
void testAttribute();
void testPragma();
void testGoto();

void testLoading() {
  testExpression();
  // testDeclaration();
  // testBlock();
  // testNamespace();
  // testTypeDecl();
  // testIf();
  // testFor();
  // testWhile();
  // testSwitch();
  // testJumps();
  // testClassAccess();
  // testAttribute();
  // testPragma();
  // testGoto();
}

void testExpression() {
  testStatementLoading("2 + 3;",
                       statementType::expression);
  testStatementLoading("-3;",
                       statementType::expression);
  testStatementLoading(";",
                       statementType::expression);
  testStatementLoading("sizeof(4);",
                       statementType::expression);
  // TODO: Test we captured the proper expression by evaluating it
}

void testDeclaration() {
}

void testBlock() {
  testStatementLoading("{}",
                       statementType::block);

  OCCA_ASSERT_EQUAL(0,
                    getStatement<blockStatement>().size());

  testStatementLoading("{\n"
                       " const int i = 0;\n"
                       " ++i:\n"
                       " namespace foo {}\n"
                       " if (true) {}\n"
                       "}\n",
                       statementType::block);

  blockStatement &smnt = getStatement<blockStatement>();
  OCCA_ASSERT_EQUAL(4,
                    smnt.size());
  OCCA_ASSERT_EQUAL_BINARY(statementType::declaration,
                           smnt[0]->type());
  OCCA_ASSERT_EQUAL_BINARY(statementType::expression,
                           smnt[1]->type());
  OCCA_ASSERT_EQUAL_BINARY(statementType::namespace_,
                           smnt[2]->type());
  OCCA_ASSERT_EQUAL_BINARY(statementType::if_,
                           smnt[3]->type());
}

void testNamespace() {
  testStatementLoading("namespace foo {}",
                       statementType::namespace_);

  OCCA_ASSERT_EQUAL("foo",
                    getStatement<namespaceStatement>().name);

  testStatementLoading("namespace A::B::C {}",
                       statementType::namespace_);

  namespaceStatement &A = getStatement<namespaceStatement>();
  OCCA_ASSERT_EQUAL("A",
                    A.name);

  namespaceStatement &B = A[0]->to<namespaceStatement>();
  OCCA_ASSERT_EQUAL("B",
                    B.name);

  namespaceStatement &C = B[0]->to<namespaceStatement>();
  OCCA_ASSERT_EQUAL("C",
                    C.name);
}

void testTypeDecl() {
  // TODO: typedef
  // TODO: struct
  // TODO: enum
  // TODO: union
  // TODO: class
}

void testIf() {
  testStatementLoading("if (true) {}",
                       statementType::if_);

  testStatementLoading("if (true) {}\n"
                       "else if (true) {}",
                       statementType::if_);

  testStatementLoading("if (true) {}\n"
                       "else if (true) {}\n"
                       "else if (true) {}",
                       statementType::if_);

  testStatementLoading("if (true) {}\n"
                       "else if (true) {}\n"
                       "else {}",
                       statementType::if_);

  // Test declaration in conditional
  testStatementLoading("if (const int i = 1) {}",
                       statementType::if_);

  // TODO: Test that 'i' exists in the if scope
}

void testFor() {
  testStatementLoading("for (;;) {}",
                       statementType::for_);
  testStatementLoading("for (;;);",
                       statementType::for_);

  // Test declaration in conditional
  testStatementLoading("for (int i = 0; i < 2; ++i) {}",
                       statementType::for_);

  // TODO: Test that 'i' exists in the if scope
}

void testWhile() {
  testStatementLoading("while (true) {}",
                       statementType::while_);
  testStatementLoading("while (true);",
                       statementType::while_);

  // Test declaration in conditional
  testStatementLoading("while (int i = 0) {}",
                       statementType::while_);

  // TODO: Test that 'i' exists in the if scope

  // Same tests for do-while
  testStatementLoading("do {} while (true);",
                       statementType::while_);
  testStatementLoading("do ; while (true);",
                       statementType::while_);

  testStatementLoading("do {} while (int i = 0)",
                       statementType::while_);
}

void testSwitch() {
  testStatementLoading("switch (2) {}",
                       statementType::switch_);
  // Weird cases
  testStatementLoading("switch (2) case 2:",
                       statementType::switch_);
  testStatementLoading("switch (2) case 2: 2;",
                       statementType::switch_);

  // Test declaration in conditional
  testStatementLoading("switch (int i = 2) {}",
                       statementType::switch_);

  // TODO: Test that 'i' exists in the if scope

  // Test caseStatement
  testStatementLoading("case 2:",
                       statementType::case_);
  testStatementLoading("case 2: 2;",
                       statementType::case_);

  // Test defaultStatement
  testStatementLoading("default:",
                       statementType::default_);
  testStatementLoading("default: 2;",
                       statementType::default_);
}

void testJumps() {
  testStatementLoading("continue;",
                       statementType::continue_);
  testStatementLoading("break;",
                       statementType::continue_);

  testStatementLoading("return;",
                       statementType::continue_);
  testStatementLoading("return 1 + (2 * 1);",
                       statementType::continue_);
  // TODO: Test 'eval' to make sure we capture the return value
}

void testClassAccess() {
  testStatementLoading("public:",
                       statementType::classAccess);
  testStatementLoading("protected:",
                       statementType::classAccess);
  testStatementLoading("private:",
                       statementType::classAccess);
}

void testAttribute() {
  testStatementLoading("@dim",
                       statementType::attribute);
  testStatementLoading("@dim(2)",
                       statementType::attribute);
  testStatementLoading("@dim(x=2, y=2)",
                       statementType::attribute);
  // TODO: Test the argument values
}

void testPragma() {
  testStatementLoading("#pragma",
                       statementType::pragma);
  testStatementLoading("#pragma occa test",
                       statementType::pragma);
  // TODO: Test the pragma source
}

void testGoto() {
  testStatementLoading("label:",
                       statementType::gotoLabel);
  testStatementLoading("goto label;",
                       statementType::goto_);
}
//======================================

//---[ Errors ]------------------------
void testExpressionErrors();
void testDeclarationErrors();
void testBlockErrors();
void testNamespaceErrors();
void testTypeDeclErrors();
void testIfErrors();
void testForErrors();
void testWhileErrors();
void testSwitchErrors();
void testJumpsErrors();
void testClassAccessErrors();
void testAttributeErrors();
void testPragmaErrors();
void testGotoErrors();

void testErrors() {
  std::cerr << "Testing parser errors:\n";

  testExpressionErrors();
  // testDeclarationErrors();
  // testBlockErrors();
  // testNamespaceErrors();
  // testTypeDeclErrors();
  // testIfErrors();
  // testForErrors();
  // testWhileErrors();
  // testSwitchErrors();
  // testJumpsErrors();
  // testClassAccessErrors();
  // testAttributeErrors();
  // testPragmaErrors();
  // testGotoErrors();
}

void testExpressionErrors() {
  parseSource("2 + 3");
  parseSource("-2");
  parseSource("2 = {}");
  parseSource("sizeof(4)");
}

void testDeclarationErrors() {
}

void testBlockErrors() {
}

void testNamespaceErrors() {
}

void testTypeDeclErrors() {
}

void testIfErrors() {
}

void testForErrors() {
}

void testWhileErrors() {
}

void testSwitchErrors() {
}

void testJumpsErrors() {
}

void testClassAccessErrors() {
}

void testAttributeErrors() {
}

void testPragmaErrors() {
}

void testGotoErrors() {
}
//======================================
