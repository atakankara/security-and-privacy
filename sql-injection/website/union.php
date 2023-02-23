<?php
include "common.php";
if (isset($_GET['source']))
    die(highlight_string(file_get_contents(__FILE__)));



$id = input("id")*1;
$username = input("username");
$query = "SELECT U.username, S.* FROM salaries S
    JOIN users U ON (U.id=S.userid)
    WHERE S.id={$id} OR U.username='{$username}'";

if ($id or $username) // An employee selected.
{
    // Run the query
    try {
        $records = sql($query);
    }
    catch (PDOException $e)
    {
        $err=$e->getMessage();
    }
    // Debug output
    echo "<pre style='background-color:#EEE;border:1px solid gray;padding:10px;'>DEBUG INFORMATION\nQuery : ", htmlspecialchars($query);
    echo PHP_EOL,str_repeat("-",80),PHP_EOL;
    if (isset($err))
        echo "Error: ", $err;
    else
        echo "Result: ", print_r($records,true);
    echo "</pre>";
    // Results
    if (empty($records))
    {
        echo "No records found.";
    }
    else
    {
        $employee = $records[0];
        if ($employee->salary > 100000) // Too high, don't show.
            $employee->salary = "Classified.";
        echo "<h1>{$employee->username}</h2>";
        echo "<div><strong>Role:</strong> {$employee->role}</div>";
        echo "<div><strong>Salary:</strong> {$employee->salary}</div>";
        echo "<div><strong>Bio:</strong> {$employee->bio}</div>";
        echo "<a href='?'>Back to List</a>";
    }
}
else // Show a list of all employees.
{
    echo "<h2>Select Employee to See Profile:</h2>";
    $users = sql("SELECT * FROM users");
    foreach ($users as $user)
        echo "<a href='?username={$user->username}'>", $user->username, "</a><br/>", PHP_EOL;
}
?>

<hr/>
<a href='?source=1' target="_blank">Source Code</a>
|
<a href='./'>Back</a>



