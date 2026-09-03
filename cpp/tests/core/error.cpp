/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/error.hpp>

#include <gtest/gtest.h>

#include <regex>
#include <source_location>
#include <string>

namespace raft {

namespace {

/** The first line of the message, i.e. the message without the appended call stack. */
auto first_line(std::string const& msg) -> std::string { return msg.substr(0, msg.find('\n')); }

auto reg_escape(std::string const& s) -> std::string
{
  static std::regex const special_chars{R"([-[\]{}()*+?.,\^$|#\s])"};
  return std::regex_replace(s, special_chars, R"(\$&)");
}

/**
 * Expected message of an error reported in this file, at a line matching @p line_pattern, in a
 * function whose name contains @p function.
 */
auto expected_message(char const* prefix,
                      std::string const& line_pattern,
                      char const* function,
                      std::string const& reason) -> std::regex
{
  std::string re{"^"};
  re += reg_escape(prefix);
  re += R"(file=.*error\.cpp line=)";
  re += line_pattern;
  re += " function=.*";
  re += function;
  re += ".*: ";
  re += reg_escape(reason);
  re += "$";
  return std::regex{re};
}

auto line_of(std::source_location location) -> std::string
{
  return std::to_string(location.line());
}

/** Any line of this file; used where the exact line is not the point of the test. */
constexpr char const* k_any_line = R"(\d+)";

/** A utility reporting an error on behalf of its caller. */
auto report_on_behalf_of_caller(std::source_location location = std::source_location::current())
  -> std::string
{
  return format_error_message(location, "RAFT failure at ", "%s", "reported by a utility");
}

}  // namespace

TEST(Error, FormatBlamesTheGivenLocation)
{
  auto const loc = std::source_location::current();
  auto msg = format_error_message(loc, "RAFT failure at ", "value=%d, name='%s'", 42, "answer");

  EXPECT_TRUE(std::regex_match(
    msg,
    expected_message(
      "RAFT failure at ", line_of(loc), "FormatBlamesTheGivenLocation", "value=42, name='answer'")))
    << "message:'" << msg << "'";
}

TEST(Error, FormatBlamesTheCallerOfAUtility)
{
  auto msg = report_on_behalf_of_caller();

  // This test function is blamed, not report_on_behalf_of_caller and not error.hpp.
  EXPECT_TRUE(std::regex_match(
    msg,
    expected_message(
      "RAFT failure at ", k_any_line, "FormatBlamesTheCallerOfAUtility", "reported by a utility")))
    << "message:'" << msg << "'";
  EXPECT_EQ(msg.find("error.hpp"), std::string::npos) << "message:'" << msg << "'";
  EXPECT_EQ(msg.find("report_on_behalf_of_caller"), std::string::npos) << "message:'" << msg << "'";
}

TEST(Error, FormatHandlesLongReasons)
{
  std::string reason{"This is a test string repeated many times. "};
  for (size_t i = 0; i < 6; ++i) {
    reason += reason;
  }
  EXPECT_TRUE(reason.size() > 2048) << "size of the test string is: " << reason.size();

  auto const loc = std::source_location::current();
  auto msg       = format_error_message(loc, "RAFT failure at ", (reason + "%d").c_str(), 121);

  EXPECT_TRUE(std::regex_match(
    msg,
    expected_message("RAFT failure at ", line_of(loc), "FormatHandlesLongReasons", reason + "121")))
    << "message:'" << msg << "'";
}

TEST(Error, ExpectsBlamesTheCallSite)
{
  int const x = -1;
  try {
    RAFT_EXPECTS(x > 0, "x must be positive, got %d", x);
    FAIL() << "Expected logic_error from a violated expectation";
  } catch (raft::logic_error const& e) {
    auto msg = first_line(e.what());
    EXPECT_TRUE(std::regex_match(
      msg,
      expected_message(
        "RAFT failure at ", k_any_line, "ExpectsBlamesTheCallSite", "x must be positive, got -1")))
      << "message:'" << msg << "'";
  }
}

TEST(Error, FailBlamesTheCallSite)
{
  try {
    RAFT_FAIL("cannot do %s", "that");
    FAIL() << "Expected logic_error from RAFT_FAIL";
  } catch (raft::logic_error const& e) {
    auto msg = first_line(e.what());
    EXPECT_TRUE(std::regex_match(
      msg,
      expected_message("RAFT failure at ", k_any_line, "FailBlamesTheCallSite", "cannot do that")))
      << "message:'" << msg << "'";
  }
}

// The deprecated macro must keep working for the projects that have not switched yet.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
TEST(Error, DeprecatedSetErrorMsgAppends)
{
  std::string msg{"prefix:"};
  ASSERT_NO_THROW(SET_ERROR_MSG(msg, "location prefix:", "value=%d", 123));

  EXPECT_TRUE(std::regex_match(
    msg,
    expected_message(
      "prefix:location prefix:", k_any_line, "DeprecatedSetErrorMsgAppends", "value=123")))
    << "message:'" << msg << "'";
}
#pragma GCC diagnostic pop

}  // namespace raft
